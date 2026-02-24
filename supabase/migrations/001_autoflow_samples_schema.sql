-- autoflow_samples DB 스키마 (Life Matrix Flow와 동일, 동일 Supabase 프로젝트 사용 가능)
-- Supabase 대시보드 SQL Editor에서 실행하거나, supabase db push 사용.

CREATE TABLE IF NOT EXISTS sample_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  file_content JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sample_input_data (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  content TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS autoflow_samples (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  app_section TEXT NOT NULL DEFAULT 'Life',
  category TEXT,
  developer_email TEXT,
  model_id UUID NOT NULL REFERENCES sample_models(id) ON DELETE CASCADE,
  input_data_id UUID REFERENCES sample_input_data(id) ON DELETE SET NULL,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE sample_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE sample_input_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE autoflow_samples ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read sample_models"
  ON sample_models FOR SELECT USING (true);
CREATE POLICY "Allow public read sample_input_data"
  ON sample_input_data FOR SELECT USING (true);
CREATE POLICY "Allow public read autoflow_samples"
  ON autoflow_samples FOR SELECT USING (true);

CREATE POLICY "Allow public insert sample_models"
  ON sample_models FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update sample_models"
  ON sample_models FOR UPDATE USING (true);
CREATE POLICY "Allow public delete sample_models"
  ON sample_models FOR DELETE USING (true);
CREATE POLICY "Allow public insert sample_input_data"
  ON sample_input_data FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update sample_input_data"
  ON sample_input_data FOR UPDATE USING (true);
CREATE POLICY "Allow public delete sample_input_data"
  ON sample_input_data FOR DELETE USING (true);
CREATE POLICY "Allow public insert autoflow_samples"
  ON autoflow_samples FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update autoflow_samples"
  ON autoflow_samples FOR UPDATE USING (true);
CREATE POLICY "Allow public delete autoflow_samples"
  ON autoflow_samples FOR DELETE USING (true);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS autoflow_samples_updated_at ON autoflow_samples;
CREATE TRIGGER autoflow_samples_updated_at
  BEFORE UPDATE ON autoflow_samples
  FOR EACH ROW EXECUTE PROCEDURE update_updated_at();

CREATE INDEX IF NOT EXISTS idx_autoflow_samples_app_section ON autoflow_samples(app_section);
CREATE INDEX IF NOT EXISTS idx_autoflow_samples_category ON autoflow_samples(category);
CREATE INDEX IF NOT EXISTS idx_autoflow_samples_created_at ON autoflow_samples(created_at DESC);
