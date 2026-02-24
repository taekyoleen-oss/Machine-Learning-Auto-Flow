/**
 * ML 샘플 6개를 Supabase에 등록하는 일회성 시드 스크립트
 * - Classification: Decision Tree, K Means, Logistic Regression
 * - Regression: Linear Regression, Linear_Reg in Boston, Stat Model
 *
 * 사용: .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY 설정 후
 *       node scripts/seed-ml-supabase-samples.js
 *       또는 pnpm run seed:supabase
 */

import dotenv from "dotenv";
import { createClient } from "@supabase/supabase-js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 프로젝트 루트(ML Auto Flow)의 .env 명시 로드 (cwd 무관)
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const APP_SECTION = "ML";

// 6개 파일, 대구분 app_section = "ML"
const SAMPLES = [
  { file: "Decision Tree.ins", name: "Decision Tree", category: "Classification" },
  { file: "K Means.ins", name: "K Means", category: "Classification" },
  { file: "Logistic Regression.ins", name: "Logistic Regression", category: "Classification" },
  { file: "Linear Regression.ins", name: "Linear Regression", category: "Regression" },
  { file: "Linear_Reg in Boston.ins", name: "Linear_Reg in Boston", category: "Regression" },
  { file: "Stat Model.ins", name: "Stat Model", category: "Regression" },
];

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
    "Missing Supabase config. Set in .env: VITE_SUPABASE_URL & VITE_SUPABASE_ANON_KEY, or NEXT_PUBLIC_SUPABASE_URL & NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY"
  );
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseAnonKey);
const samplesDir = path.join(__dirname, "..", "samples");

async function seed() {
  console.log("Seeding ML samples to Supabase...\n");

  const { data: existingRows } = await supabase
    .from("autoflow_samples")
    .select("id, category, sample_models!inner ( name )")
    .eq("app_section", APP_SECTION);
  const existingSet = new Set(
    (existingRows || []).map((r) => `${r.sample_models?.name ?? ""}|${r.category ?? ""}`)
  );
  console.log(`기존 ML 샘플 ${existingRows?.length ?? 0}개 확인됨. 없으면 추가합니다.\n`);

  for (const { file, name, category } of SAMPLES) {
    const key = `${name}|${category}`;
    if (existingSet.has(key)) {
      console.log(`⏭  이미 있음: ${name} (${category})`);
      continue;
    }
    const filePath = path.join(samplesDir, file);
    if (!fs.existsSync(filePath)) {
      console.warn(`⚠  File not found: ${file}`);
      continue;
    }

    let content;
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch (e) {
      console.error(`❌ Read failed: ${file}`, e.message);
      continue;
    }

    let data;
    try {
      data = JSON.parse(content);
    } catch (e) {
      console.error(`❌ Invalid JSON: ${file}`, e.message);
      continue;
    }

    const modules = Array.isArray(data.modules) ? data.modules : [];
    const connections = Array.isArray(data.connections) ? data.connections : [];
    const file_content = { modules, connections };

    try {
      const { data: modelRow, error: modelError } = await supabase
        .from("sample_models")
        .insert({ name, file_content })
        .select("id")
        .single();

      if (modelError) {
        console.error(`❌ sample_models insert failed: ${name}`, modelError.message);
        continue;
      }

      const modelId = modelRow.id;
      const { error: sampleError } = await supabase.from("autoflow_samples").insert({
        app_section: APP_SECTION,
        category,
        developer_email: null,
        description: null,
        model_id: modelId,
        input_data_id: null,
      });

      if (sampleError) {
        console.error(`❌ autoflow_samples insert failed: ${name}`, sampleError.message);
        continue;
      }

      existingSet.add(key);
      console.log(`✅ ${name} (${category})`);
    } catch (e) {
      console.error(`❌ ${name}`, e.message);
    }
  }

  console.log("\nDone.");
}

seed();
