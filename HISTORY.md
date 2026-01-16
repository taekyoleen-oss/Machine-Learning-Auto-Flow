# Change History

## 2026-01-16 (현재 작업)

### feat(samples): DB 기반 샘플 관리 시스템 구현

**Description:**
- SQLite 데이터베이스를 사용한 샘플 관리 시스템 구현
- Samples 버튼 클릭 시 팝업 모달로 카드 형식 샘플 표시
- 샘플 관리 모달에서 파일 업로드, 수정, 삭제 기능 추가
- Express API 서버에 샘플 관리 엔드포인트 추가
- 기존 파일 기반 샘플을 DB로 마이그레이션하는 스크립트 추가

**Files Affected:**
- `components/SamplesModal.tsx` - 샘플 카드 형식 모달 컴포넌트 (신규)
- `components/SamplesManagementModal.tsx` - 샘플 관리 모달 컴포넌트 (신규)
- `utils/samples-api.ts` - 샘플 API 클라이언트 (신규)
- `server/db/samples-db.js` - SQLite 데이터베이스 초기화 모듈 (신규)
- `server/routes/samples.js` - 샘플 관리 API 라우터 (신규)
- `scripts/migrate-samples-to-db.js` - 샘플 마이그레이션 스크립트 (신규)
- `samples/samples-metadata.json` - 샘플 메타데이터 파일 (신규)
- `samples/FILE_FORMAT_GUIDE.md` - 샘플 파일 형식 가이드 (신규)
- `App.tsx` - DB에서 샘플 로드하도록 수정, 관리 모달 통합
- `server/split-data-server.js` - 샘플 API 라우터 연결
- `scripts/build-samples.js` - 메타데이터 병합 로직 추가
- `package.json` - better-sqlite3 패키지 추가, migrate:samples 스크립트 추가
- `README.md` - 앱 실행 방법 및 시스템 구성 설명 추가

**Reason:**
- 샘플을 파일 기반에서 DB 기반으로 전환하여 관리 용이성 향상
- 샘플 추가/수정/삭제 기능을 UI로 제공하여 사용자 편의성 개선
- 카드 형식 UI로 샘플 정보를 더 명확하게 표시

**Commit Hash:** (커밋 후 업데이트)

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-13 (이전 작업)

### fix(vercel): Add package-lock.json and restore npm ci command

**Description:**
- `package-lock.json` 파일 생성하여 Vercel 빌드 시 `npm ci` 오류 해결
- `npm install --package-lock-only` 명령으로 `package-lock.json` 생성
- `vercel.json`의 `installCommand`를 `npm ci --prefer-offline --no-audit`로 복원
- 이제 `npm ci`가 정상적으로 작동하여 더 빠르고 안정적인 빌드 가능

**Files Affected:**
- `package-lock.json` - 새로 생성 (4,415줄)
- `vercel.json` - `installCommand`를 `npm ci --prefer-offline --no-audit`로 복원

**Reason:**
- Vercel 빌드 시 `npm ci` 명령이 `package-lock.json`을 요구하여 빌드 실패
- `package-lock.json`을 생성하여 `npm ci`가 정상 작동하도록 수정
- `npm ci`는 `npm install`보다 빠르고 재현 가능한 빌드를 제공

**Commit Hash:** 179aad7

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 179aad7

# Or direct recovery
git reset --hard 179aad7
```

### fix(vercel): Change installCommand from npm ci to npm install

**Description:**
- Vercel 빌드 오류 해결: `package-lock.json`이 없어 `npm ci` 명령이 실패하는 문제 수정
- `vercel.json`의 `installCommand`를 `npm ci`에서 `npm install`로 변경
- `package-lock.json`이 없어도 정상적으로 빌드되도록 수정

**Files Affected:**
- `vercel.json` - `installCommand`를 `npm install`로 변경

**Reason:**
- Vercel 빌드 시 `npm ci` 명령이 `package-lock.json`을 요구하여 빌드 실패
- `npm install`은 `package-lock.json`이 없어도 작동하므로 빌드 안정성 향상

**Commit Hash:** b8a7d91

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b8a7d91

# Or direct recovery
git reset --hard b8a7d91
```

### chore: Trigger Vercel redeployment to update Statistics Dtype display

**Description:**
- Vercel 재배포를 트리거하여 Statistics 모듈의 Dtype 행 표시 업데이트
- 빈 커밋을 생성하여 Vercel 자동 배포 프로세스 시작
- Statistics 모듈의 View Details에서 Dtype 행이 올바르게 표시되도록 보장

**Files Affected:**
- 없음 (빈 커밋)

**Reason:**
- Vercel 배포 사이트(https://www.insureautoflow.com/)에 최신 코드가 반영되지 않은 것으로 확인
- Statistics 모듈의 View Details에서 Dtype 행이 표의 첫 부분에 올바르게 표시되도록 재배포 필요
- 최신 커밋(42937f3)의 변경사항이 Vercel에 반영되도록 재배포 트리거

**Commit Hash:** 5c954df

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 5c954df

# Or direct recovery
git reset --hard 5c954df
```

## 2026-01-12 (이전 작업)

### refactor(git): Create new Git repository to remove api folder history

**Description:**
- 기존 Git 저장소와의 연결을 끊고 새로운 GitHub 저장소에 푸시
- Git 히스토리에서 `api` 폴더 관련 기록 완전 제거
- Vercel 빌드 캐시 문제 해결을 위한 깨끗한 시작
- 새 저장소: `https://github.com/taekyoleen-oss/Machine-Learning-Auto-Flow.git`
- 기존 저장소: `https://github.com/taekyoleen-oss/ML_Auto_Flow.git` (연결 해제)

**Files Affected:**
- `.git/` - 기존 Git 히스토리 삭제 후 새로 초기화
- 모든 프로젝트 파일 - 새 저장소에 초기 커밋으로 추가 (179개 파일, 102,572줄)

**Reason:**
- Git 히스토리에 남아있던 `api` 폴더로 인한 Vercel 빌드 오류 해결
- Vercel 빌드 캐시에서 존재하지 않는 `api` 폴더를 참조하는 문제 방지
- 깨끗한 Git 히스토리로 프로젝트 재시작

**Commit Hash:** 08b49ca

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 08b49ca

# Or direct recovery
git reset --hard 08b49ca
```

## 2026-01-12 (이전 작업)

### refactor(ui): Move 'Save as Sample' button to Samples menu top

**Description:**
- "Sample로 저장" 버튼을 툴바에서 Samples 메뉴 상단으로 이동
- Samples 드롭다운 메뉴를 열면 최상단에 "Sample로 저장" 버튼 표시
- 버튼 클릭 시 메뉴 자동 닫힘
- 모듈이 없으면 disabled 상태로 표시

**Files Affected:**
- `App.tsx` - "Sample로 저장" 버튼을 Samples 메뉴 내부로 이동, 툴바에서 제거

**Reason:**
- Samples 관련 기능을 Samples 메뉴 내에 통합하여 UI 일관성 향상
- 사용자가 Samples 메뉴에서 바로 Sample을 저장할 수 있도록 개선

**Commit Hash:** 15d098a

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### feat(samples): Add client-side Sample and Example creation without server

**Description:**
- 서버 없이 브라우저에서 직접 Sample과 Example을 생성할 수 있는 기능 추가
- App.tsx에 "Sample로 저장" 버튼 추가: 현재 파이프라인을 Sample JSON 파일로 다운로드
- PropertiesPanel.tsx에 "Example로 저장" 버튼 추가: Load Data 모듈의 현재 데이터를 CSV 파일로 다운로드
- 다운로드된 파일을 `samples` 또는 `Examples_in_Load` 폴더에 복사하면 빌드 시 자동으로 포함됨
- 서버 의존성 없이 완전히 클라이언트 사이드에서 작동

**Files Affected:**
- `App.tsx` - `handleSaveAsSample` 함수 추가 및 툴바에 "Sample로 저장" 버튼 추가
- `components/PropertiesPanel.tsx` - `handleSaveAsExample` 함수 추가 및 Examples 섹션에 "Example로 저장" 버튼 추가
- `components/icons.tsx` - `ArrowDownTrayIcon` import 확인 (이미 존재)

**Reason:**
- 서버 없이 사용자가 직접 Sample과 Example을 생성할 수 있도록 개선
- 개발 워크플로우 간소화: 파일 다운로드 → 폴더에 복사 → 빌드
- 모든 브라우저에서 작동하며 파일 시스템 접근 권한 불필요

**Commit Hash:** bbd06cc

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### refactor(server): Remove Samples Server and keep only Split Data and PPT generation APIs

**Description:**
- Split Data API와 PPT 생성 API만 서버에 유지하고 나머지 서버 기능 제거
- `server/samples-server.js` 파일 전체 삭제
- `server/split-data-server.js`에서 Samples API 관련 코드 제거 (Split Data와 PPT 생성만 유지)
- `package.json`에서 `samples-server` 스크립트 제거 및 `dev` 스크립트 수정
- `vite.config.ts`에서 Samples, Examples-in-Load 프록시 제거 (PPT 생성은 유지)
- Samples와 Examples는 이미 정적 JSON 파일(`/samples.json`, `/examples-in-load.json`)로 직접 로드하므로 서버 불필요
- AI 기능들(AI로 파이프라인 생성하기, AI로 데이터 분석 실행하기)은 브라우저에서 직접 Gemini API 호출하므로 서버 불필요

**Files Affected:**
- `server/samples-server.js` - 삭제
- `server/split-data-server.js` - Samples API 관련 코드 제거 (라인 25-31, 161-424), Split Data API와 PPT 생성 API만 유지
- `package.json` - `samples-server` 스크립트 제거, `dev` 스크립트에서 `samples-server` 제거
- `vite.config.ts` - `/api/samples`, `/api/examples-in-load` 프록시 제거

**Reason:**
- 서버 기능 최소화로 프로젝트 단순화
- Samples와 Examples는 이미 정적 JSON 파일로 작동하므로 서버 불필요
- AI 기능들은 브라우저에서 직접 Gemini API 호출하므로 서버 불필요
- Split Data와 PPT 생성만 Python 스크립트 실행이 필요하므로 서버 유지

**Commit Hash:** 00fbd5d

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Add .vercelignore and exclude api folders from TypeScript

**Description:**
- `.vercelignore` 파일 생성하여 `api` 폴더를 Vercel 빌드에서 제외
- `tsconfig.json`에서 `api` 및 `pages/api` 폴더를 명시적으로 제외
- Vercel 빌드 캐시 문제로 인한 TypeScript 오류 해결을 위한 추가 설정
- Git 저장소에는 `api` 폴더가 없지만 Vercel이 이전 빌드 캐시를 사용할 수 있어 명시적 제외 설정 추가

**Files Affected:**
- `.vercelignore` - 새 파일: `api` 폴더를 Vercel 빌드에서 제외
- `tsconfig.json` - `api`, `**/api/**`, `pages/api`, `**/pages/api/**` exclude에 추가
- `vercel.json` - 불필요한 `functions` 설정 제거

**Reason:**
- Vercel 빌드 시 `api/samples/list.ts`와 `api/samples/[filename].ts`에서 `@vercel/node` 모듈을 찾을 수 없다는 TypeScript 오류 해결
- Vercel 빌드 캐시로 인해 존재하지 않는 `api` 폴더를 참조하는 문제 방지
- TypeScript가 `api` 폴더를 체크하지 않도록 명시적 제외 설정

**Commit Hash:** 5130b03

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### refactor: Complete removal of server dependencies - use JSON files only

**Description:**
- 서버 기능을 완전히 제거하고 JSON 파일만 사용하도록 재설계
- `pages/api` 폴더 전체 삭제 (Vercel Serverless Functions)
- 사용되지 않는 유틸리티 파일 삭제 (`utils/samples.ts`, `shared/utils/samples.ts`)
- 모든 데이터를 빌드 시점에 생성된 JSON 파일에서 직접 로드
- 서버 없이 완전히 정적 파일로 작동하도록 완전히 전환

**Files Affected:**
- `pages/api/` 폴더 전체 삭제 (split-data.ts, samples/)
- `utils/samples.ts` 삭제
- `shared/utils/samples.ts` 삭제

**Reason:**
- Vercel 빌드 시 `pages/api` 폴더의 TypeScript 오류 해결
- 서버 의존성 완전 제거로 프로젝트 단순화
- 빌드 시점에 모든 데이터를 JSON 파일로 생성하여 런타임에 정적 파일만 사용
- Vercel 배포 시 서버 없이 완전히 작동하도록 보장

**Commit Hash:** b7cb583

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### refactor(vercel): Remove unused API folder and @vercel/node package

**Description:**
- 더 이상 사용하지 않는 `api` 폴더 전체 삭제
- Vercel Serverless Functions가 필요 없으므로 `@vercel/node` 패키지 제거
- `vercel.json`에서 `/api/(.*)` rewrites 제거
- `tsconfig.json`에서 `api` exclude 제거
- 빌드 시점에 JSON 파일을 생성하는 방식으로 완전히 전환하여 API 서버 불필요

**Files Affected:**
- `api/` 폴더 전체 삭제 (samples, examples-in-load)
- `package.json` - `@vercel/node` 패키지 제거
- `vercel.json` - API rewrites 제거
- `tsconfig.json` - api exclude 제거

**Reason:**
- 빌드 시점에 JSON 파일을 생성하는 방식으로 완전히 전환하여 API 서버가 더 이상 필요 없음
- Vercel 빌드 시 `@vercel/node` 관련 TypeScript 오류 해결
- 불필요한 코드와 의존성 제거로 프로젝트 단순화

**Commit Hash:** ee26696

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Move @vercel/node back to dependencies and exclude api folder from tsconfig

**Description:**
- `@vercel/node` 패키지를 다시 `dependencies`로 이동
- Vercel 빌드 시점에 TypeScript 타입 체크를 위해 `dependencies`에 필요
- `tsconfig.json`에서 `api` 폴더를 제외하여 클라이언트 빌드 시 API 파일 타입 체크 건너뛰기
- Vercel Serverless Functions는 런타임에 `@vercel/node`를 자동으로 제공하므로 실제로는 설치되지 않음

**Files Affected:**
- `package.json` - `@vercel/node`를 `dependencies`로 이동
- `tsconfig.json` - `api` 폴더를 exclude에 추가

**Reason:**
- Vercel 빌드 시 `devDependencies`가 설치되지 않아 TypeScript 오류 발생
- `dependencies`에 포함하여 빌드 시점 타입 체크 통과
- `api` 폴더를 `tsconfig.json`에서 제외하여 클라이언트 빌드와 분리

**Commit Hash:** e8da6ae

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Move @vercel/node to devDependencies for Vercel build

**Description:**
- `@vercel/node` 패키지를 `dependencies`에서 `devDependencies`로 이동
- Vercel Serverless Functions는 런타임에 `@vercel/node`를 자동으로 제공하므로 타입 정의만 필요
- 빌드 시점에 TypeScript 타입 체크를 위해 `devDependencies`에 포함

**Files Affected:**
- `package.json` - `@vercel/node`를 `devDependencies`로 이동

**Reason:**
- Vercel 빌드 시 `@vercel/node` 모듈을 찾을 수 없다는 TypeScript 오류 해결
- Vercel Serverless Functions는 런타임에 자동으로 제공되므로 타입 정의만 빌드 시점에 필요
- `devDependencies`에 포함하여 빌드 시점 타입 체크는 통과하되 런타임 의존성은 추가하지 않음

**Commit Hash:** bc02145

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Add @vercel/node package to fix TypeScript build errors

**Description:**
- Vercel 빌드 시 `@vercel/node` 모듈을 찾을 수 없다는 TypeScript 오류 해결
- `package.json`의 `dependencies`에 `@vercel/node` 패키지 추가
- Vercel Serverless Functions의 TypeScript 타입 정의 제공

**Files Affected:**
- `package.json` - `@vercel/node` 패키지를 dependencies에 추가

**Reason:**
- Vercel 빌드 시 `api/samples/list.ts`와 `api/samples/[filename].ts`에서 `@vercel/node` 모듈을 찾을 수 없는 오류 해결
- Vercel Serverless Functions를 위한 필수 타입 정의 패키지 추가

**Commit Hash:** 3a9c395

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Use routes instead of rewrites for JSON file serving in Vercel

**Description:**
- Vercel 배포 환경에서 JSON 파일이 여전히 로드되지 않는 문제 해결
- `vercel.json`에서 `rewrites` 대신 `routes`를 사용하도록 변경
- `routes`는 `rewrites`보다 우선순위가 높아 정적 파일 처리에 더 적합
- JSON 파일을 명시적으로 처리하도록 `routes`에 추가
- JSON 파일에 대한 캐시 및 Content-Type 헤더 설정 추가
- `cleanUrls`와 `trailingSlash` 설정 추가

**Files Affected:**
- `vercel.json` - `rewrites`를 `routes`로 변경, JSON 파일 명시적 처리, 헤더 설정 추가

**Reason:**
- Vercel 배포 시 Samples와 Examples-in-Load JSON 파일이 여전히 로드되지 않는 문제 해결
- `rewrites`가 모든 경로를 가로채는 문제를 `routes`의 우선순위를 활용하여 해결
- 정적 파일이 먼저 처리되도록 보장

**Commit Hash:** 6d92994

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Fix JSON file routing in vercel.json for static file serving

**Description:**
- Vercel 배포 환경에서 `/samples.json`과 `/examples-in-load.json` 파일이 로드되지 않는 문제 해결
- `vercel.json`의 `rewrites` 설정에서 JSON 파일을 명시적으로 처리하도록 수정
- SPA 라우팅을 위한 `/(.*)` 패턴이 정적 파일도 `index.html`로 리다이렉트하는 문제 해결
- JSON 파일을 먼저 처리하도록 `rewrites` 순서 조정

**Files Affected:**
- `vercel.json` - JSON 파일을 명시적으로 처리하도록 rewrites 추가 및 순서 조정

**Reason:**
- Vercel 배포 시 Samples와 Examples-in-Load JSON 파일이 로드되지 않는 문제 해결
- SPA 라우팅 설정이 정적 파일 요청도 가로채는 문제를 해결하기 위해 JSON 파일을 명시적으로 처리

**Commit Hash:** 968861b

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### refactor(build): Replace server-based file loading with build-time JSON generation

**Description:**
- Samples와 Examples-in-Load 파일 로딩을 서버 기반에서 빌드 시점 JSON 생성 방식으로 변경
- 서버 없이 완전히 정적 파일로 작동하도록 개선
- Vercel 배포 환경에서도 안정적으로 작동하도록 수정
- 빌드 스크립트(`scripts/build-samples.js`) 생성하여 samples와 Examples_in_Load 폴더의 파일들을 JSON으로 변환
- 클라이언트 코드를 수정하여 `/samples.json`과 `/examples-in-load.json`을 직접 fetch하도록 변경
- .mla 파일 형식 지원 제거 (.ins와 .json만 지원)

**Files Affected:**
- `scripts/build-samples.js` - 새 파일: 빌드 시점에 samples와 Examples_in_Load 폴더의 파일들을 JSON으로 변환
- `package.json` - build 스크립트에 `build-samples.js` 실행 추가
- `App.tsx` - API 호출 대신 `/samples.json` 직접 fetch, .mla 파일 형식 제외
- `components/PropertiesPanel.tsx` - API 호출 대신 `/examples-in-load.json` 직접 fetch, getApiBaseUrl 함수 제거
- `api/samples/list.ts` - .mla 파일 형식 처리 제거
- `api/samples/[filename].ts` - .mla 파일 형식 처리 제거
- `utils/fileOperations.ts` - 파일 선택 다이얼로그에서 .mla 제거
- `utils/samples.ts` - 주석에서 .mla 예시를 .ins로 변경
- `shared/utils/fileOperations.ts` - 파일 선택 다이얼로그에서 .mla 제거
- `shared/utils/samples.ts` - 주석에서 .mla 예시를 .ins로 변경

**Reason:**
- Vercel 배포 환경에서 Samples와 Examples가 로드되지 않는 문제 해결
- 서버 의존성 제거로 더 안정적이고 간단한 아키텍처 구현
- 빌드 시점에 모든 데이터를 정적 파일로 변환하여 런타임 성능 향상
- .mla 파일 형식은 더 이상 사용하지 않으므로 제거

**Commit Hash:** e2a24b5

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Add Examples-in-Load API Serverless Functions for Vercel deployment

**Description:**
- Load Data 모듈의 EXAMPLES가 Vercel 배포 환경에서 작동하지 않는 문제 해결
- Examples_in_Load 폴더의 CSV 파일을 읽기 위한 Vercel Serverless Functions 생성
- Samples API와 동일한 패턴으로 경로 해결 로직 구현
- PropertiesPanel에서 환경별 API URL 동적 설정 (getApiBaseUrl 함수 추가)
- 프로덕션에서는 상대 경로, 개발 환경에서는 localhost 사용

**Files Affected:**
- `api/examples-in-load/list.ts` - 새 파일: Examples_in_Load 폴더의 CSV 파일 목록 반환
- `api/examples-in-load/[filename].ts` - 새 파일: 특정 CSV 파일 내용 반환
- `components/PropertiesPanel.tsx` - getApiBaseUrl 함수 추가, Examples-in-Load API 호출 로직 개선

**Reason:**
- Vercel 배포 시 Load Data 모듈의 EXAMPLES가 로드되지 않는 문제 해결
- Samples API와 동일한 방식으로 Examples-in-Load API도 Vercel Serverless Functions로 구현
- 프로덕션 환경에서도 Examples_in_Load 폴더의 CSV 파일들이 정상적으로 로드되도록 수정

**Commit Hash:** 8fe1e01

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Improve Samples API path resolution for Vercel deployment

**Description:**
- Vercel Serverless Functions에서 `samples` 폴더 경로를 찾지 못하는 문제 해결
- `import.meta.url`을 사용하여 ESM 모듈에서 `__dirname` 획득
- 여러 가능한 경로를 순차적으로 시도하도록 개선 (process.cwd(), __dirname 기반 상대 경로, 절대 경로)
- 경로를 찾지 못할 경우 상세한 디버깅 로그 출력
- Vercel 배포 환경에서도 Samples API가 정상 작동하도록 경로 해결 로직 개선

**Files Affected:**
- `api/samples/list.ts` - 경로 해결 로직 개선, ESM 모듈 호환성 추가
- `api/samples/[filename].ts` - 경로 해결 로직 개선, 에러 처리 및 디버깅 로그 추가

**Reason:**
- Vercel 배포 시 `samples` 폴더를 찾지 못하여 Samples 목록이 비어있거나 파일을 읽을 수 없는 문제 해결
- 다양한 Vercel 배포 환경에서도 안정적으로 작동하도록 경로 해결 로직 강화

**Commit Hash:** b794514

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-12 (이전 작업)

### fix(vercel): Fix Vercel deployment issues - remove local network permission request and enable Samples API

**Description:**
- Vercel 배포 시 로컬 네트워크 권한 요청 메시지 제거
- 프로덕션 빌드에서 `host: '0.0.0.0'` 설정 제거 (개발 환경에서만 적용)
- Samples API가 프로덕션에서 작동하도록 Vercel Serverless Functions 생성
- 환경에 따라 API URL 동적 설정 (개발: localhost, 프로덕션: 상대 경로)

**Files Affected:**
- `vite.config.ts` - 프로덕션 빌드 시 host 설정 조건부 적용
- `App.tsx` - `getApiBaseUrl()` 함수 추가하여 환경별 API URL 동적 설정
- `api/samples/list.ts` - Vercel Serverless Function (샘플 목록 조회)
- `api/samples/[filename].ts` - Vercel Serverless Function (특정 샘플 파일 읽기)
- `vercel.json` - API 라우트 설정 추가

**Reason:**
- Vercel 배포 시 로컬 네트워크 권한 요청이 나타나는 문제 해결
- 도메인 설정 후 Samples 폴더의 예제가 보이지 않는 문제 해결
- 프로덕션 환경에서 Samples API가 정상 작동하도록 수정

**Commit Hash:** 7b42e79

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 7b42e79

# Or direct recovery
git reset --hard 7b42e79
```

## 2026-01-09 (이전 작업)

### fix(connections): Add null checks for connection objects to prevent runtime errors

**Description:**
- 연결 객체(connection)와 from/to 속성에 대한 null 체크 추가
- `Cannot read properties of null (reading 'id')` 에러 수정
- Optional chaining을 사용하여 fromPortName 접근 시 안전성 향상
- MortalityResult 모듈 실행 시 연결 객체 null 체크 추가
- getSingleInputData, handleRearrangeModules 등 연결 관련 함수들에 null 체크 추가

**Files Affected:**
- `App.tsx` - 연결 객체 null 체크 추가 (MortalityResult, getSingleInputData, handleRearrangeModules, areUpstreamModulesReady 등)
- `components/Canvas.tsx` - getConnectionData 함수에 null 체크 추가
- `components/DataPreviewModal.tsx` - 입력 연결 찾기 시 null 체크 추가
- `components/PropertiesPanel.tsx` - getConnectedDataSourceHelper 함수에 null 체크 추가
- `components/ClusteringDataPreviewModal.tsx` - trainedModel과 originalData useMemo에 null 체크 추가

**Reason:**
- 연결 객체가 null이거나 from/to 속성이 없는 경우 발생하는 런타임 에러 방지
- 애플리케이션 안정성 향상

**Commit Hash:** 7282543

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 7282543

# Or direct recovery
git reset --hard 7282543
```

## 2026-01-09 (이전 작업)

### feat(mortality): Add mortality forecasting models (Lee-Carter, CBD, APC, RH, Plat, P-Spline) and comparison module

**Description:**
- Advanced Model 카테고리 추가 (Tradition Analysis 다음에 위치)
- Mortality Model 서브카테고리 추가
- 6개의 사망률 예측 모델 모듈 추가:
  - Lee-Carter Model: log(m_x,t) = a_x + b_x * k_t
  - CBD Model: logit(q_x,t) = k_t^(1) + (x - x_bar) * k_t^(2)
  - APC Model: log(m_x,t) = a_x + k_t + g_c
  - RH Model: log(m_x,t) = a_x + b_x^(1) * k_t^(1) + b_x^(2) * k_t^(2) + b_x^(3) * g_t-x
  - Plat Model: Lee-Carter + CBD 결합 모델
  - P-Spline Model: 스플라인 기반 모델
- Mortality Result 모듈 추가: 여러 모델 결과를 비교하고 시각화
- 각 모델은 Pyodide를 통해 Python으로 실행
- 모델 비교 시 사망률 곡선, 시간 추세, 메트릭 비교, 예측 비교 시각화 제공

**Files Affected:**
- `types.ts` - ModuleType enum에 7개 모듈 타입 추가, MortalityModelOutput 및 MortalityResultOutput 인터페이스 추가
- `constants.ts` - TOOLBOX_MODULES에 7개 모듈 정의 추가, DEFAULT_MODULES에 기본 파라미터 추가
- `components/Toolbox.tsx` - Advanced Model 카테고리 및 Mortality Model 서브카테고리 추가
- `data_analysis_modules.py` - 6개 모델 피팅 함수 및 compare_mortality_models 함수 추가
- `utils/pyodideRunner.ts` - 각 모델별 Python 실행 함수 추가
- `App.tsx` - 각 모델 실행 로직 및 MortalityResult 실행 로직 추가

**Reason:**
- 생명보험/액추어리 분야에서 사용하는 사망률 예측 모델을 Pyodide를 통해 브라우저에서 실행 가능하도록 구현
- 여러 모델을 비교하여 최적의 모델을 선택할 수 있는 기능 제공

**Commit Hash:** e669610

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-09 (이전 작업)

### fix(preprocessing): Fix Prep Normalize View Details parameter display and Evaluate Model input data retrieval

**Description:**
- Prep Normalize 모듈의 View Details 첫 번째 탭에서 파라미터 값이 표시되지 않는 문제 수정
- 입력 데이터 가져오기 로직 개선 (다양한 출력 타입 처리, SplitDataOutput 지원)
- inputData.columns가 배열이 아닌 경우 처리 추가
- Evaluate Model의 입력 데이터 가져오기 수정 (포트 이름 명시, scored_data_out 지원)
- getSingleInputData 함수 개선 (다양한 출력 포트 이름 지원, 디버깅 로그 추가)
- PropertiesPanel에 NormalizeData 모듈의 columnSelections 초기화 로직 추가
- View Details의 모든 콘텐츠에서 텍스트 복사(Ctrl+C) 가능하도록 userSelect CSS 추가
- PrepNormalizeProcessingInfo 컴포넌트의 로직을 App.tsx와 일치시켜 일관성 확보

**Files Affected:**
- `App.tsx` - Evaluate Model 입력 데이터 가져오기 수정, getSingleInputData 함수 개선, NormalizeData 실행 로직 개선
- `components/DataPreviewModal.tsx` - PrepNormalizeProcessingInfo 컴포넌트 개선, 입력 데이터 처리 로직 수정, 텍스트 복사 기능 추가
- `components/PropertiesPanel.tsx` - NormalizeData 모듈의 columnSelections 초기화 로직 추가

**Reason:**
- Prep Normalize 모듈의 파라미터 정보를 사용자가 확인할 수 있도록 개선
- Evaluate Model이 Score Model의 출력을 올바르게 받을 수 있도록 수정
- View Details의 데이터를 복사하여 다른 곳에 활용할 수 있도록 개선

**Commit Hash:** 95bd05d

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-09 (이전 작업)

### feat(clustering): Add comprehensive View Details for clustering modules

**Description:**
- Train Clustering Model 모듈의 View Details 구현 (K-Means 및 PCA)
- Clustering Data 모듈의 View Details 구현 (K-Means 및 PCA)
- K-Means: 클러스터 중심점, 품질 지표, 클러스터별 통계량, 클러스터 간 거리, 실루엣 점수, Davies-Bouldin Index 추가
- PCA: 설명된 분산 비율, Scree Plot, 주성분 공간 시각화, 재구성 오차, 주성분 계수 행렬 추가
- 적합 수준 통계량을 상단에 배치하여 모델 성능을 한눈에 확인 가능하도록 개선
- K-Means Clustering 모듈의 독립 실행 버튼 제거 (다른 지도학습 모듈과 동일하게 처리)

**Files Affected:**
- `components/TrainedClusteringModelPreviewModal.tsx` - 새 파일: Train Clustering Model 전용 Preview Modal
- `components/ClusteringDataPreviewModal.tsx` - 새 파일: Clustering Data 전용 Preview Modal
- `App.tsx` - ClusteringDataPreviewModal 및 TrainedClusteringModelPreviewModal 연결, viewingClusteringData state 추가
- `components/DataPreviewModal.tsx` - ClusteringDataOutput 및 TrainedClusteringModelOutput 처리 제거 (별도 모달로 분리)
- `components/ComponentRenderer.tsx` - K-Means 모듈의 실행 버튼 제거 (noRunButtonTypes에 추가)

**Reason:**
- 클러스터링 모델의 성능 평가를 위한 상세 통계량 제공
- K-Means와 PCA의 적합 수준을 시각적으로 확인 가능하도록 개선
- 다른 모듈의 View Details에 영향을 주지 않도록 별도 모달로 분리

**Commit Hash:** f0f3983

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard <커밋해시>

# Or direct recovery
git reset --hard <커밋해시>
```

## 2026-01-09 11:16:04

### feat(layout): Improve Auto Layout with parent-child level guarantee and model definition positioning

**Description:**
- 모델 정의 모듈(Linear Regression 등)을 같은 레벨의 위쪽에 배치하도록 정렬 로직 추가
- Train Model의 경우 모델 정의 모듈과 데이터 입력 모듈을 같은 컬럼에 세로 배치하고 Train Model을 중간에 배치
- 모든 입력 모듈이 출력 모듈보다 최소 한 컬럼 왼쪽에 배치되도록 보장하는 로직 추가
- 2입력 모듈 처리 후에도 부모-자식 레벨 관계가 유지되도록 최종 검증 로직 추가

**Files Affected:**
- `App.tsx` - handleRearrangeModules 함수 수정
  - 레벨 그룹화 후 모델 정의 모듈을 위쪽에 배치하도록 정렬 로직 추가 (3.5단계)
  - Train Model의 부모 모듈들을 같은 컬럼에 배치하고 Train Model을 중간에 배치하는 로직 개선
  - 모든 연결에 대해 부모가 자식보다 최소 한 레벨 앞에 있도록 보장하는 로직 추가 (2.6, 2.8단계)
  - 2입력 모듈 처리 후 최종 부모-자식 레벨 검증 로직 추가

**Reason:**
- 모듈 간 연결이 겹치지 않도록 보장하여 레이아웃 가독성 향상
- 데이터 흐름이 왼쪽에서 오른쪽으로 명확하게 표시되도록 개선
- 모델 정의 모듈과 데이터 입력 모듈의 관계를 시각적으로 명확하게 표현

**Commit Hash:** fabe596

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard fabe596

# Or direct recovery
git reset --hard fabe596
```

## 2026-01-08 14:20:08

### feat(layout): Improve Auto Layout for modules with 2 inputs

**Description:**
- 입력이 2개인 모듈의 경우 첫 번째 입력이 두 번째 입력보다 위에 위치하도록 수정
- 두 입력 모듈을 동일한 레벨에 배치하도록 수정
- 이미 배치된 모듈(Split Data 등)은 위치를 변경하지 않도록 수정
- modulesWithTwoInputs 변수 정의 순서 문제 해결 (Auto Layout 버그 수정)

**Files Affected:**
- `App.tsx` - handleRearrangeModules 함수 수정
  - modulesWithTwoInputs 정의를 레벨 계산 직후로 이동
  - 포트 이름 우선순위 기반 입력 순서 결정 (data_in > model_in > others)
  - 두 부모 모듈의 레벨을 동일하게 조정하는 로직 추가
  - 이미 배치된 모듈(placedModules)은 재조정하지 않도록 수정

**Reason:**
- 입력이 2개인 모듈의 레이아웃을 더 직관적으로 개선
- 첫 번째 입력이 항상 위에 위치하도록 보장하여 가독성 향상
- Split Data처럼 이미 사용된 모듈의 위치를 보존하여 기존 레이아웃 유지

**Commit Hash:** 71e936c

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 71e936c

# Or direct recovery
git reset --hard 71e936c
```

## 2026-01-02 15:19:03

### fix(build): Fix import paths in App.tsx for Vercel deployment

**Description:**
- App.tsx에서 shared 폴더 import 경로 수정 (../shared → ./shared)
- Vercel 빌드 시 발생하던 "Could not resolve ../shared/utils/fileOperations" 오류 해결
- fileOperations와 samples 유틸리티 import 경로 수정

**Files Affected:**
- `App.tsx` - import 경로 수정 (../shared/utils/fileOperations → ./shared/utils/fileOperations, ../shared/utils/samples → ./shared/utils/samples)

**Reason:**
- App.tsx가 루트 디렉토리에 있고 shared 폴더도 루트 디렉토리에 있어서 상대 경로가 잘못되었음
- Vercel 빌드 환경에서 경로 해석 오류로 인한 빌드 실패 해결

**Commit Hash:** 371aceb

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 7d36254

# Or direct recovery
git reset --hard 7d36254
```

## 2025-01-XX XX:XX:XX

### feat(modules): Update Random Forest module and fix Decision Tree visualization

**Description:**
- Random Forest 모듈 파라미터 레이블 변경 (N Estimators → n_estimators, Max Depth → max_depth)
- Random Forest 모듈에 max_features 파라미터 추가 (기본값: None - 전체 특징 사용)
- PropertiesPanel에 max_features UI 추가 (None, auto, sqrt, log2, 커스텀 숫자 입력 옵션)
- data_analysis_modules.py의 create_random_forest 함수에 max_features 파라미터 지원 추가
- Decision Tree View Details 시각화 수정: TrainedModelOutput 인터페이스에 trainingData와 modelParameters 필드 추가

**Files Affected:**
- `types.ts` - TrainedModelOutput 인터페이스에 trainingData와 modelParameters 필드 추가
- `constants.ts` - Random Forest 모듈에 max_features 파라미터 추가
- `components/PropertiesPanel.tsx` - Random Forest 파라미터 레이블 변경 및 max_features UI 추가
- `data_analysis_modules.py` - create_random_forest 함수에 max_features 파라미터 추가

**Reason:**
- Random Forest 모듈 파라미터 이름을 Python 스타일로 통일
- max_features 파라미터를 통해 각 트리에서 고려할 특징 수를 제어할 수 있도록 기능 확장
- Decision Tree 모듈의 View Details에서 트리 시각화가 정상적으로 표시되도록 수정

**Commit Hash:** b5fdcfe

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b5fdcfe

# Or direct recovery
git reset --hard b5fdcfe
```

## 2025-01-XX XX:XX:XX

### feat(modules): Add Column Plot module with comprehensive chart visualization

**Description:**
- Column Plot 모듈 추가: 단일열 또는 2개열 선택을 통한 다양한 차트 시각화 기능
- 컬럼 데이터 타입(숫자형/범주형)에 따라 자동으로 적절한 차트 옵션 제공
- View Details 모달에서 차트 타입 선택 및 실시간 차트 생성
- matplotlib 기반 Python 차트 생성 (seaborn 의존성 제거)
- 지원하는 차트 타입:
  - 단일열 숫자형: Histogram, KDE Plot, Boxplot, Violin Plot, ECDF Plot, QQ-Plot, Line Plot, Area Plot
  - 단일열 범주형: Bar Plot, Count Plot, Pie Chart, Frequency Table
  - 2개열 숫자형+숫자형: Scatter Plot, Hexbin Plot, Joint Plot, Line Plot, Regression Plot, Heatmap
  - 2개열 숫자형+범주형: Box Plot, Violin Plot, Bar Plot, Strip Plot, Swarm Plot
  - 2개열 범주형+범주형: Grouped Bar Plot, Heatmap, Mosaic Plot

**Files Affected:**
- `types.ts` - ModuleType.ColumnPlot 및 ColumnPlotOutput 타입 추가
- `constants.ts` - Column Plot 모듈 정의 추가
- `components/Toolbox.tsx` - Column Plot 모듈을 Toolbox에 추가
- `components/PropertiesPanel.tsx` - Column Plot 속성 UI 구현 (단일열/2개열 선택)
- `components/ColumnPlotPreviewModal.tsx` - View Details 모달 생성 (새 파일)
- `utils/pyodideRunner.ts` - createColumnPlotPython 함수 추가 (matplotlib만 사용)
- `App.tsx` - Column Plot 실행 로직 및 모달 연결 추가

**Reason:**
- 데이터 시각화 기능 확장
- 사용자가 데이터를 다양한 방식으로 시각화할 수 있도록 지원
- 컬럼 타입에 따른 자동 차트 옵션 제공으로 사용자 편의성 향상

**Commit Hash:** 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf

# Or direct recovery
git reset --hard 1b7ea562b784a8fbd28a3e8041efd74f6ee6d8cf
```

## 2025-12-14 09:35:26

### feat(pca): Improve PCA visualization with JavaScript-based implementation

**Description:**
- PCA 시각화를 JavaScript 기반(ml-pca)으로 전환하여 성능 개선
- Python Pyodide 의존성 제거로 안정성 향상
- Label Column을 선택 사항으로 변경하고 Predict를 기본값으로 설정
- 이진 분류(0/1)를 위한 간소한 색상 체계 적용 (파란색: 클래스 0, 빨간색: 클래스 1)
- Color Scale 범례 제거 및 그래프 너비 1400px로 확장
- 콤보박스에서 "None (Basic PCA)" 옵션 제거
- 그래프 가시성 개선 (그리드 라인, 축 레이블, 레이아웃 개선)

**Files Affected:**
- `utils/pcaCalculator.ts` - ml-pca 라이브러리를 사용한 JavaScript 기반 PCA 계산 함수 추가
- `components/DataPreviewModal.tsx` - PCA Visualization 개선 (Label Column 선택 사항화, 그래프 크기 및 스타일 개선)
- `package.json` - ml-pca 라이브러리 의존성 추가
- `pnpm-lock.yaml` - 의존성 업데이트

**Reason:**
- Python Pyodide 기반 PCA 구현에서 발생한 패키지 로딩 및 데이터 마샬링 문제 해결
- 브라우저 환경에서 더 안정적이고 빠른 PCA 계산을 위해 JavaScript 기반 구현으로 전환
- 사용자 경험 개선을 위한 시각화 개선

**Commit Hash:** de7bb9092853b58ba903cf6788e0904a2c4d05d7

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7

# Or direct recovery
git reset --hard de7bb9092853b58ba903cf6788e0904a2c4d05d7
```

## 2025-12-12 17:50:00

### feat(samples): Add Samples folder support and Linear Regression-1 sample

**Description:**
- Samples 폴더 기능 추가 및 Linear Regression-1 샘플 추가
- Samples 폴더의 파일을 자동으로 읽어서 Samples 메뉴에 표시하는 기능 구현
- Save 버튼으로 저장한 .mla 파일을 samples 폴더에 넣으면 자동으로 표시되도록 개선
- File System Access API 오류 처리 개선
- 파일 이름의 공백 및 특수문자 처리 (URL 인코딩/디코딩)

**Files Affected:**
- `App.tsx` - Samples 폴더 파일 로드 기능 추가, File System Access API 오류 처리 개선
- `server/samples-server.js` - Samples 폴더 파일 목록 및 읽기 API 구현
- `savedSamples.ts` - Linear Regression-1 샘플 추가
- `samples/README.md` - Samples 폴더 사용 방법 문서 추가
- `samples/example.json` - 예제 파일 추가
- `package.json` - samples-server 스크립트 추가
- `vite.config.ts` - /api/samples 프록시 설정 추가
- `types.ts` - StatsModelFamily에 Logit, QuasiPoisson 추가, DiversionCheckerOutput, EvaluateStatOutput 타입 추가

**Reason:**
- 사용자가 Save 버튼으로 저장한 모델을 samples 폴더에 넣으면 자동으로 Samples 메뉴에 표시되도록 하기 위해
- Linear Regression-1 샘플을 공유 가능한 샘플로 추가

**Commit Hash:** b7dfe9fc6c744f5d41e2d417afa575205c80fbec

**Recovery Command:**
```bash
# Backup and recover
git stash push -u -m "백업"
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec

# Or direct recovery
git reset --hard b7dfe9fc6c744f5d41e2d417afa575205c80fbec
```
