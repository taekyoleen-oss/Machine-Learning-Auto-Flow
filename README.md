<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# ML Auto Flow - 머신러닝 파이프라인 빌더

머신러닝 워크플로우를 시각적으로 구성하고 실행할 수 있는 웹 애플리케이션입니다.

## 시스템 요구사항

- **Node.js** (v18 이상 권장)
- **Python** (3.8 이상, 일부 기능 사용 시)
- **npm** 또는 **pnpm**

## 설치 방법

1. **의존성 설치**:
   ```bash
   npm install
   ```

2. **환경 변수 설정** (선택사항):
   `.env.local` 파일을 생성하고 Gemini API 키를 설정하세요:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **샘플 데이터 마이그레이션** (최초 1회):
   ```bash
   npm run migrate:samples
   ```
   기존 파일 기반 샘플을 데이터베이스로 이전합니다.

## 실행 방법

### 방법 1: 개발 모드 (권장) - 한 번에 실행

프론트엔드와 백엔드 서버를 **동시에** 실행합니다:

```bash
npm run dev
```

이 명령어는 다음을 실행합니다:
- **백엔드 서버** (포트 3002): 샘플 관리 API, SplitData API, PPT 생성 API
- **프론트엔드** (포트 3003): Vite 개발 서버 (DFA가 3001 사용 시 충돌 방지)

실행 후 브라우저에서 `http://localhost:3003`로 접속하세요.

### 방법 2: 개별 실행

#### 프론트엔드만 실행
```bash
npm run dev:client-only
```
- 포트 3003에서 실행
- 백엔드 서버 없이도 기본 기능 사용 가능 (샘플 관리 기능 제외)

#### 백엔드 서버만 실행
```bash
npm run server
```
- 포트 3002에서 실행
- 샘플 관리, SplitData, PPT 생성 API 제공

## 서버 구성

애플리케이션은 두 개의 서버로 구성됩니다:

### 1. 프론트엔드 서버 (Vite)
- **포트**: 3003
- **역할**: React 애플리케이션 제공
- **URL**: `http://localhost:3003`

### 2. 백엔드 서버 (Express)
- **포트**: 3002
- **역할**: 
  - **SplitData API** (`/api/split-data`) — 데이터 분할, 서버 기동 시 항상 사용 가능
  - **PPT 생성 API** (`/api/generate-ppts`) — PowerPoint 생성
  - **Samples API** (`/api/samples`) — 샘플 목록/CRUD, better-sqlite3 로드 시 사용 가능(미로드 시 503)
- **URL**: `http://localhost:3002`
- 서버 시작 시 콘솔에 각 API별 `(ready)` / Samples DB 미사용 시 `(DB unavailable, 503)` 상태가 출력됩니다.

### Samples API (`/api/samples`) 필요 여부

| 환경 | Samples API 필요 여부 |
|------|------------------------|
| **Supabase 설정됨** (`.env`에 `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`) | **불필요** — 목록·단건 로드·등록·수정·삭제 모두 Supabase 사용 |
| **Supabase 미설정** | **필요** — 샘플 관리 모달 목록/수정/삭제 및 숫자 ID로 샘플 로드 시 서버 API 사용 (better-sqlite3 필요) |

Supabase를 사용 중이면 `better-sqlite3` 없이 서버가 떠 있어도 Samples 기능은 정상 동작합니다. 404/CSP 관련 콘솔 메시지 중 `localhost:3002/.well-known/appspecific/com.chrome.devtools.json` 은 Chrome DevTools 내부 요청으로, 무시해도 됩니다.

## 주요 기능

### 샘플 관리
- **Samples 버튼**: 샘플 목록을 카드 형식으로 표시
- **관리 버튼**: 샘플 추가, 수정, 삭제
- **데이터베이스**: SQLite를 사용한 샘플 저장

### 데이터 처리
- 데이터 로드 및 전처리
- 머신러닝 모델 학습
- 모델 평가 및 예측

### PPT 생성
- 파이프라인을 PowerPoint로 자동 생성

## 빌드 및 배포

### 프로덕션 빌드
```bash
npm run build
```

빌드된 파일은 `dist` 폴더에 생성됩니다.

### 미리보기
```bash
npm run preview
```

## 문제 해결

### 포트가 이미 사용 중인 경우

**에러**: `EADDRINUSE: address already in use :::3002`

**해결 방법**:
1. Windows PowerShell에서:
   ```powershell
   # 포트 3002를 사용하는 프로세스 찾기
   netstat -ano | findstr :3002
   
   # 프로세스 종료 (PID는 위 명령어 결과에서 확인)
   Stop-Process -Id <PID> -Force
   ```

2. 또는 다른 포트 사용:
   ```bash
   # 환경 변수로 포트 변경
   $env:SERVER_PORT=3003
   npm run server
   ```

### 샘플 관리 기능이 작동하지 않는 경우

1. **Supabase 사용 시** (권장):
   - `.env`에 `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` 설정
   - 서버의 Samples API는 사용하지 않으므로 better-sqlite3/서버 없이도 Samples 목록·실행·등록 가능

2. **Supabase 미사용 시 (로컬 DB)**:
   - 서버 실행: `npm run server`
   - better-sqlite3 필요 시: `pnpm rebuild better-sqlite3`
   - `database/samples.db` 확인, 없으면 `npm run migrate:samples` 실행

3. **브라우저 콘솔 확인**:
   - F12 → Network 탭에서 API 요청 확인
   - `localhost:3002/.well-known/...` 404/CSP 메시지는 Chrome DevTools 내부 요청으로 무시 가능

### 샘플이 표시되지 않는 경우

1. **마이그레이션 실행**:
   ```bash
   npm run migrate:samples
   ```

2. **샘플 파일 확인**:
   - `samples/` 폴더에 `.ins` 또는 `.json` 파일이 있는지 확인

## 프로젝트 구조

```
ML Auto Flow/
├── components/          # React 컴포넌트
├── server/              # Express 백엔드 서버
│   ├── db/             # 데이터베이스 모듈
│   └── routes/         # API 라우터
├── samples/            # 샘플 파일
├── database/           # SQLite 데이터베이스
├── public/             # 정적 파일
└── utils/              # 유틸리티 함수
```

## 추가 정보

- **샘플 관리 가이드**: `samples/README.md`
- **파일 형식 가이드**: `samples/FILE_FORMAT_GUIDE.md`
- **PPT 생성 가이드**: `PPT_생성_가이드.md`

## 라이선스

이 프로젝트는 개인 사용을 위한 것입니다.
