/// <reference types="vite/client" />

// The app bundle is typechecked WITHOUT @types/node (see tsconfig "types": ["vite/client"]),
// because @types/node's built-in iterator globals break `Array.from(...)` overload resolution
// on Set/Map/FileList under this TS/lib combination. A couple of bundle files still reference
// `process.env` for a guarded, dev-only API-key fallback that Vite statically replaces at build
// time. This ambient, type-only declaration keeps those guarded accesses type-safe. It emits no
// JavaScript and does not alter runtime behavior.
declare const process: {
  env: Record<string, string | undefined>;
};
