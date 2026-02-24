/**
 * Supabase에 등록된 ML 샘플 개수·목록 확인
 * 사용: node scripts/check-ml-supabase-samples.js
 */

import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { createClient } from "@supabase/supabase-js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const supabaseUrl =
  process.env.VITE_SUPABASE_URL ||
  process.env.NEXT_PUBLIC_SUPABASE_URL ||
  process.env.SUPABASE_URL ||
  "";
const supabaseAnonKey =
  process.env.VITE_SUPABASE_ANON_KEY ||
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY ||
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
  process.env.SUPABASE_ANON_KEY ||
  "";

if (!supabaseUrl || !supabaseAnonKey) {
  console.error(
    "Missing Supabase config. Set in .env: NEXT_PUBLIC_SUPABASE_URL & NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY (or VITE_*)"
  );
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function check() {
  const { data: rows, error } = await supabase
    .from("autoflow_samples")
    .select(`
      id,
      category,
      created_at,
      sample_models!inner ( name )
    `)
    .eq("app_section", "ML")
    .order("created_at", { ascending: false });

  if (error) {
    console.error("Supabase 조회 실패:", error.message);
    process.exit(1);
  }

  const list = rows || [];
  console.log(`\n[ML] Supabase 등록 샘플: 총 ${list.length}개\n`);
  if (list.length === 0) {
    console.log("  (없음). 등록: pnpm run seed:supabase\n");
    return;
  }
  list.forEach((r, i) => {
    const name = r.sample_models?.name ?? "-";
    console.log(`  ${i + 1}. ${name}  (${r.category ?? "-"})`);
  });
  console.log("");
}

check();
