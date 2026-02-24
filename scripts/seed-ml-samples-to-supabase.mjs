/**
 * ML Auto Flow samples.json을 Supabase에 등록하는 스크립트
 *
 * 사용법: node scripts/seed-ml-samples-to-supabase.mjs
 *
 * .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY (또는 NEXT_PUBLIC_*) 가 있어야 합니다.
 * public/samples.json을 읽어 app_section "ML"로 등록합니다.
 */

import { createClient } from "@supabase/supabase-js";
import { readFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

function loadEnv() {
  const envPath = resolve(projectRoot, ".env");
  if (!existsSync(envPath)) {
    console.error(".env 파일을 찾을 수 없습니다.");
    process.exit(1);
  }
  const content = readFileSync(envPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      const eq = trimmed.indexOf("=");
      if (eq > 0) {
        const key = trimmed.slice(0, eq).trim();
        const value = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
        process.env[key] = value;
      }
    }
  }
}

loadEnv();

const supabaseUrl =
  process.env.VITE_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseKey =
  process.env.VITE_SUPABASE_ANON_KEY ||
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error(
    "Supabase URL/Key가 없습니다. .env에 VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY를 설정하세요."
  );
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

const samplesPath = resolve(projectRoot, "public", "samples.json");
if (!existsSync(samplesPath)) {
  console.error("public/samples.json을 찾을 수 없습니다.");
  process.exit(1);
}

const raw = readFileSync(samplesPath, "utf-8");
let items;
try {
  items = JSON.parse(raw);
} catch (e) {
  console.error("samples.json 파싱 실패:", e.message);
  process.exit(1);
}

if (!Array.isArray(items)) {
  console.error("samples.json은 배열이어야 합니다.");
  process.exit(1);
}

async function main() {
  console.log(`ML 샘플 ${items.length}건 등록 시작 (app_section: ML)`);

  for (let i = 0; i < items.length; i++) {
    const s = items[i];
    const name = s.name || s.filename || `Sample ${i + 1}`;
    const data = s.data;
    if (!data || !Array.isArray(data.modules)) {
      console.warn(`[${i + 1}] "${name}" 건너뜀: data.modules 없음`);
      continue;
    }

    const modules = data.modules;
    const connections = Array.isArray(data.connections) ? data.connections : [];

    let inputDataId = null;
    const loadDataModule = modules.find((m) => m.type === "LoadData");
    if (loadDataModule?.parameters?.fileContent) {
      const content = loadDataModule.parameters.fileContent;
      const inputName = s.inputData || `${name}_input`;
      const { data: inputRow, error: inputErr } = await supabase
        .from("sample_input_data")
        .insert({ name: inputName, content: content })
        .select("id")
        .single();
      if (!inputErr && inputRow) {
        inputDataId = inputRow.id;
      }
    }

    const { data: modelRow, error: modelErr } = await supabase
      .from("sample_models")
      .insert({ name, file_content: { modules, connections } })
      .select("id")
      .single();

    if (modelErr) {
      console.error(`[${i + 1}] "${name}" sample_models 삽입 실패:`, modelErr.message);
      continue;
    }

    const { error: sampleErr } = await supabase.from("autoflow_samples").insert({
      app_section: "ML",
      category: s.category || "머신러닝",
      developer_email: null,
      model_id: modelRow.id,
      input_data_id: inputDataId,
      description: s.description || null,
    });

    if (sampleErr) {
      console.error(`[${i + 1}] "${name}" autoflow_samples 삽입 실패:`, sampleErr.message);
      continue;
    }

    console.log(`  [${i + 1}/${items.length}] ${name}`);
  }

  console.log("\n완료. ML Auto Flow Samples 메뉴에서 확인하세요.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
