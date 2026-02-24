# ML Auto Flow Supabase Samples

Samples 데이터는 Life Matrix Flow / DFA-Auto-Flow와 **동일한 Supabase 프로젝트**의 `autoflow_samples` 스키마를 사용합니다.

- **대분류(app_section)**: `ML`
- 스키마 적용: `supabase/migrations/001_autoflow_samples_schema.sql` (이미 적용된 경우 생략)

## 시드 실행

`.env`에 `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`(또는 `NEXT_PUBLIC_*`) 설정 후:

```bash
node scripts/seed-ml-samples-to-supabase.mjs
```

또는: `pnpm run seed:samples`
