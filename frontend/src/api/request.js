const API_BASE = typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE_URL
  ? import.meta.env.VITE_API_BASE_URL
  : "";
const TOKEN_KEY = "reviewop-token";

export function authHeaders(token) {
  const headers = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

export async function request(path, options = {}) {
  const storedToken = typeof localStorage !== "undefined" ? localStorage.getItem(TOKEN_KEY) : "";
  const headers = { ...(options.headers || {}) };
  if (storedToken && !headers.Authorization && !headers.authorization) {
    headers.Authorization = `Bearer ${storedToken}`;
  }
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    throw new Error("Backend is unreachable. Start backend API and check VITE_PROXY_TARGET/VITE_API_BASE_URL.");
  }

  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = text || null;
  }

  if (!response.ok) {
    const detail = payload?.detail;
    let message = payload?.error || payload;
    if (Array.isArray(detail)) {
      message = detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
    } else if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object") {
      message = detail.msg || JSON.stringify(detail);
    }
    if (typeof message !== "string") {
      message = `Request failed: ${response.status}`;
    }
    throw new Error(message);
  }

  return payload;
}

export async function downloadFile(path, filename) {
  const storedToken = typeof localStorage !== "undefined" ? localStorage.getItem(TOKEN_KEY) : "";
  const headers = storedToken ? authHeaders(storedToken) : {};
  const response = await fetch(`${API_BASE}${path}`, { headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Download failed: ${response.status}`);
  }
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}
