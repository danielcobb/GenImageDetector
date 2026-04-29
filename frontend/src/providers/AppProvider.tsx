import React, { useState, useEffect, useContext, type ReactNode } from "react";
import { AppContext, type HistoryItem } from "../contexts/AppContext";
import { AuthContext } from "../contexts/AuthContext";

const STORAGE_KEY = "anonymous_history";

const isQuotaError = (err: unknown): boolean =>
  err instanceof DOMException &&
  (err.name === "QuotaExceededError" ||
    err.name === "NS_ERROR_DOM_QUOTA_REACHED");

// localStorage caps around 5 MB per origin. Each history item holds a base64
// data URL of the analyzed image, which can easily be 2-4 MB. If saving the
// full history overflows the quota, drop the oldest items first, then fall
// back to stripping image blobs so at least the metadata survives.
const persistAnonymousHistory = (history: HistoryItem[]): void => {
  const tryWrite = (items: HistoryItem[]): boolean => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
      return true;
    } catch (err) {
      if (!isQuotaError(err)) throw err;
      return false;
    }
  };

  if (tryWrite(history)) return;

  for (let n = history.length - 1; n > 0; n--) {
    if (tryWrite(history.slice(0, n))) {
      console.warn(
        `[history] localStorage quota exceeded; kept ${n} of ${history.length} items`
      );
      return;
    }
  }

  const stripped = history.map((item, idx) =>
    idx === 0 ? item : { ...item, image: undefined }
  );
  if (tryWrite(stripped)) {
    console.warn(
      "[history] localStorage quota exceeded; dropped image previews from older items"
    );
    return;
  }

  console.warn("[history] localStorage quota exceeded; clearing persisted history");
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
};

export const AppProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentResult, setCurrentResult] = useState<HistoryItem | null>(null);
  const authContext = useContext(AuthContext);

  // Reset state on logout
  useEffect(() => {
    const handleLogout = () => {
      setCurrentResult(null);
      setHistory([]);
    };

    window.addEventListener("auth:logout", handleLogout);
    return () => window.removeEventListener("auth:logout", handleLogout);
  }, []);

  // Load history on mount - either from localStorage (anonymous) or backend (logged in)
  useEffect(() => {
    if (authContext?.loading) return;

    if (authContext?.user && authContext?.token) {
      // User is logged in - fetch history from backend
      fetchHistoryFromBackend(authContext.token);
    } else {
      // Anonymous user - load from localStorage
      loadLocalHistory();
    }
  }, [authContext?.user, authContext?.token, authContext?.loading]);

  // Save to localStorage for anonymous users whenever history changes
  useEffect(() => {
    if (authContext?.loading) return;
    if (authContext?.user) return;

    persistAnonymousHistory(history);
  }, [history, authContext?.user, authContext?.loading]);

  const loadLocalHistory = () => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setHistory(parsed);
      }
    } catch (err) {
      console.error("Failed to load history from localStorage:", err);
    }
  };

  const fetchHistoryFromBackend = async (token: string) => {
    try {
      const res = await fetch("http://localhost:8000/history", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (res.ok) {
        const data = await res.json();
        // Transform backend format to frontend format
        const transformedHistory: HistoryItem[] = data.map((item: any) => ({
          id: item.id,
          image: item.image_data,
          filename: item.filename,
          results: item.model_results.map((mr: any) => ({
            model: mr.model_name,
            confidence: mr.confidence,
          })),
          analysis: {
            model: "Aggregate",
            confidence: item.aggregate_confidence,
          },
          timestamp: item.created_at,
        }));
        setHistory(transformedHistory);
      }
    } catch (err) {
      console.error("Failed to fetch history from backend:", err);
    }
  };

  const refreshHistory = async () => {
    if (authContext?.user && authContext?.token) {
      await fetchHistoryFromBackend(authContext.token);
    }
  };

  return (
    <AppContext.Provider
      value={{
        currentResult,
        setCurrentResult,
        history,
        setHistory,
        refreshHistory,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};
