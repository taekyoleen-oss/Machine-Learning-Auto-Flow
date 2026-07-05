/** @type {import('tailwindcss').Config} */
// CDN(cdn.tailwindcss.com) → PostCSS 빌드 통합.
// darkMode는 기존 인라인 설정(index.html)과 동일하게 'class'.
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./contexts/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
    "./utils/**/*.{ts,tsx}",
  ],
  // 동적 조립 클래스 방어(PropertiesPanel `text-${highlight.color}-400` —
  // 현재 color 값이 실제로 설정되는 곳은 없으나 향후 사용 대비).
  safelist: [
    "text-green-400",
    "text-red-400",
    "text-yellow-400",
    "text-blue-400",
    "text-purple-400",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
