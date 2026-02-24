import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl =
  import.meta.env.VITE_SUPABASE_URL ||
  import.meta.env.NEXT_PUBLIC_SUPABASE_URL ||
  "";
const supabaseAnonKey =
  import.meta.env.VITE_SUPABASE_ANON_KEY ||
  import.meta.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY ||
  "";

const configured = Boolean(supabaseUrl && supabaseAnonKey);
if (!configured) {
  console.warn(
    "[Supabase] URL 또는 Anon Key가 없습니다. .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY를 설정하세요."
  );
}

/** URL/Anon Key가 설정된 경우에만 생성됨. 미설정 시 null이라 Supabase 기능은 비활성화됩니다. */
export const supabase: SupabaseClient | null = configured
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

export function isSupabaseConfigured(): boolean {
  return configured;
}
