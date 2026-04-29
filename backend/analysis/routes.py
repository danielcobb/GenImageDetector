"""Analysis API routes for image upload and analysis."""
import io
import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from PIL import Image
from sqlalchemy.orm import Session

from db.database import get_db
from analysis.models import Analysis, ModelResult
from analysis.schemas import AnalysisResponse, HistoryMigrationRequest, HistoryMigrationResponse
from auth.models import User
from auth.routes import get_current_user
from ml.classifiers.base import AIvsHumanClassifier, NYUADClassifier
from ml.classifiers.cnnspot import CNNSpotClassifier
from ml.classifiers.effort import EffortClassifier
from ml.classifiers.npr import NPRClassifier
from ml.classifiers.demo import DemoClassifier

router = APIRouter(tags=["Analysis"])

# Initialize classifiers
cnnspot_classifier = CNNSpotClassifier(
    "ml/models/CNNSpot/2025_10_22_epoch_best.pth",
    crop_size=224,
    quiet=True
)
effort_classifier = EffortClassifier(
    "ml/models/Effort/effort_clip_L14_trainOn_sdv14.pth",
    quiet=True,
)
npr_classifier = NPRClassifier(
    "ml/models/NPR/NPR_GenImage_sdv4.pth",
    quiet=True,
)
ai_vs_human_classifier = AIvsHumanClassifier()
nyuad_classifier = NYUADClassifier()
nebula_comb_v3_classifier = DemoClassifier(seed="nebula")
open_x8100_classifier = DemoClassifier(seed="quasar")


@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(None),
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze image for AI generation. Supports both:
    - Multipart file upload (from frontend)
    - JSON with base64 image data (from batch scripts)

    If user is authenticated, saves analysis to history.
    """
    img = None
    filename = ""
    image_base64 = None

    # Check if it's a JSON request with base64 image
    if file is None:
        try:
            body = await request.json()
            if "image" in body:
                # Decode base64 image
                img_data = body["image"]
                if img_data.startswith("data:image"):
                    # Remove data URL prefix
                    image_base64 = img_data  # Store full data URL for database
                    img_data = img_data.split(",", 1)[1]
                else:
                    image_base64 = f"data:image/png;base64,{img_data}"
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                filename = body.get("filename", "")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON or base64 image: {e}")
    else:
        # Handle file upload
        filename = file.filename or ""
        img_bytes = await file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Convert to base64 for storage
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if img is None:
        raise HTTPException(status_code=400, detail="No image provided")

    # Run analysis
    results = {
        "AI_vs_Human": ai_vs_human_classifier.analyze(img),
        "CNNSpot": cnnspot_classifier.analyze(img),
        "Effort": effort_classifier.analyze(img),
        "Nebula_comb_v3": nebula_comb_v3_classifier.analyze(img, filename=filename),
        "NPR": npr_classifier.analyze(img),
        "NYUAD": nyuad_classifier.analyze(img),
        "open-X8100": open_x8100_classifier.analyze(img, filename=filename),
    }

    # Calculate aggregate (same as frontend)
    confidences = list(results.values())
    weights = [abs(c - 50) for c in confidences]
    total_weight = sum(weights)
    weighted_sum = sum(c * w for c, w in zip(confidences, weights))
    aggregate_confidence = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0

    # Save to database if user is authenticated
    analysis_id = None
    if current_user:
        analysis = Analysis(
            user_id=current_user.id,
            filename=filename or "untitled",
            image_data=image_base64,
            aggregate_confidence=aggregate_confidence
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        analysis_id = analysis.id

        # Add model results
        for model_name, confidence in results.items():
            model_result = ModelResult(
                analysis_id=analysis.id,
                model_name=model_name,
                confidence=confidence
            )
            db.add(model_result)
        db.commit()

    return {"analysis_id": analysis_id, "results": results}


@router.get("/history", response_model=list[AnalysisResponse])
async def get_history(current_user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's analysis history."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).order_by(Analysis.created_at.desc()).all()
    return analyses


@router.post("/migrate-history", response_model=HistoryMigrationResponse)
async def migrate_history(
    request: HistoryMigrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Migrate anonymous user history to authenticated user account.
    Accepts a batch of analyses with existing results and saves them.
    """
    migrated_count = 0
    failed_count = 0

    for item in request.items:
        try:
            # Save analysis to database
            analysis = Analysis(
                user_id=current_user.id,
                filename=item.filename or "migrated-image",
                image_data=item.image,
                aggregate_confidence=item.aggregate_confidence
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)

            # Add model results
            for result in item.model_results:
                model_result = ModelResult(
                    analysis_id=analysis.id,
                    model_name=result.model_name,
                    confidence=result.confidence
                )
                db.add(model_result)
            db.commit()

            migrated_count += 1

        except Exception as e:
            failed_count += 1
            print(f"Failed to migrate item {item.filename}: {e}")
            continue

    return HistoryMigrationResponse(
        migrated_count=migrated_count,
        failed_count=failed_count,
        message=f"Successfully migrated {migrated_count} items, {failed_count} failed"
    )
