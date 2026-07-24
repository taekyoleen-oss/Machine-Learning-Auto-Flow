# Samples 폴더 파일 형식 가이드

이 문서는 `samples` 폴더에 추가할 수 있는 파일 형식을 정확히 설명합니다.

## 지원 파일 확장자

- `.ins` - 표준 확장자(권장). 앱의 "저장" 버튼도 `.ins`로 저장합니다. 모든 샘플 파일은 `.ins`로 통일합니다.
- `.json` - 하위 호환용으로 계속 지원(기존 파일 로드용, 신규 파일은 `.ins` 사용).

빌드 스크립트(`scripts/build-samples.js`)는 **확장자가 아니라 파일 내용**(연결이 `from.moduleId` 형식인지)으로 아래 두 형식을 자동 판별하므로, 어떤 확장자를 쓰든 두 구조 중 하나만 맞으면 정상 동작합니다.

## 형식 1: id 기반 연결 (.ins 파일 — Save 버튼으로 저장한 형식)

### 필수 구조

```json
{
  "name": "프로젝트 이름",  // 또는 "projectName" 필드 사용 가능
  "modules": [
    {
      "id": "module-1768455356970-0",  // 필수: 각 모듈의 고유 ID
      "type": "LoadData",
      "name": "데이터 로드",
      "position": { "x": 50, "y": 50 },
      "parameters": {
        "source": "BostonHousing.csv"
      }
    }
    // ... 더 많은 모듈
  ],
  "connections": [
    {
      "from": {
        "moduleId": "module-1768455356970-0",  // 필수: 모듈의 id 참조
        "portName": "data_out"
      },
      "to": {
        "moduleId": "module-1768455356970-1",  // 필수: 모듈의 id 참조
        "portName": "data_in"
      }
    }
    // ... 더 많은 연결
  ]
}
```

### 특징
- 각 모듈은 **고유한 `id`** 필드를 가져야 합니다
- 연결(`connections`)은 모듈의 `id`를 참조합니다 (`moduleId`)
- 빌드 스크립트가 자동으로 `id` 기반 연결을 `fromModuleIndex/toModuleIndex` 형식으로 변환합니다
- 프로젝트 이름은 `name` 또는 `projectName` 필드에서 가져옵니다

### 예시 파일
- `Linear_Reg in Boston.ins`
- `Linear Regression.ins`
- `Decision Tree.ins`
- `KMeans-Insurance.ins`

## 형식 2: 인덱스 기반 연결 (모듈 배열 인덱스 사용)

### 필수 구조

```json
{
  "name": "Linear Regression",
  "modules": [
    {
      "type": "LoadData",
      "name": "데이터 로드",
      "position": { "x": 50, "y": 50 },
      "parameters": {
        "source": "your-data-source.csv"
      }
    }
    // ... 더 많은 모듈
  ],
  "connections": [
    {
      "fromModuleIndex": 0,  // 필수: 모듈 배열의 인덱스 (0부터 시작)
      "fromPort": "data_out",
      "toModuleIndex": 1,   // 필수: 모듈 배열의 인덱스 (0부터 시작)
      "toPort": "data_in"
    }
    // ... 더 많은 연결
  ]
}
```

### 특징
- 모듈에 `id` 필드가 **없습니다**
- 연결(`connections`)은 모듈 배열의 **인덱스**를 사용합니다 (`fromModuleIndex`, `toModuleIndex`)
- 인덱스는 0부터 시작합니다
- 프로젝트 이름은 `name` 필드에서 가져옵니다

### 예시 파일
- `Book_AdultIncome_DecisionTree.ins`
- `Book_Wholesale_KMeans.ins`

(과거엔 이 형식을 `.json` 확장자로 저장했으나, 이제 확장자는 `.ins`로 통일하고 내용 구조로만 판별합니다.)

## 모듈 필수 필드

두 형식 모두 다음 필드를 가져야 합니다:

```json
{
  "type": "ModuleType",      // 필수: 모듈 타입 (예: "LoadData", "TrainModel")
  "name": "모듈 이름",        // 선택: 모듈 이름 (없으면 type 사용)
  "position": {              // 선택: 위치 (없으면 {x: 0, y: 0} 사용)
    "x": 50,
    "y": 50
  },
  "parameters": {}           // 선택: 모듈 파라미터 (없으면 빈 객체)
}
```

## 연결 필수 필드

### .ins 형식의 연결
```json
{
  "from": {
    "moduleId": "module-id-string",  // 필수: 모듈의 id
    "portName": "data_out"            // 필수: 포트 이름
  },
  "to": {
    "moduleId": "module-id-string",   // 필수: 모듈의 id
    "portName": "data_in"             // 필수: 포트 이름
  }
}
```

### .json 형식의 연결
```json
{
  "fromModuleIndex": 0,      // 필수: 모듈 배열의 인덱스
  "fromPort": "data_out",    // 필수: 포트 이름
  "toModuleIndex": 1,        // 필수: 모듈 배열의 인덱스
  "toPort": "data_in"       // 필수: 포트 이름
}
```

## 파일 생성 방법

### 방법 1: 앱에서 Save 버튼 사용 (권장)
1. 앱에서 파이프라인을 구성합니다
2. 상단의 "Save" 버튼을 클릭합니다
3. `.ins` 파일로 저장됩니다
4. 저장된 파일을 `samples` 폴더에 복사합니다

### 방법 2: 수동으로 JSON 파일 작성
1. 위의 형식에 맞춰 JSON 파일을 작성합니다
2. `.json` 또는 `.ins` 확장자로 저장합니다
3. `samples` 폴더에 저장합니다

## 빌드 프로세스

빌드 시 (`npm run build`):
1. `scripts/build-samples.js`가 실행됩니다
2. `samples` 폴더의 `.json`과 `.ins` 파일을 읽습니다
3. `.ins` 파일의 `id` 기반 연결을 `fromModuleIndex/toModuleIndex`로 변환합니다
4. `public/samples.json` 파일을 생성합니다
5. 앱에서 `/samples.json`을 통해 샘플 목록을 로드합니다

## 주의사항

1. **파일 확장자**: `.json` 또는 `.ins`만 지원됩니다
2. **JSON 유효성**: 파일은 유효한 JSON 형식이어야 합니다
3. **모듈 ID**: `.ins` 파일의 경우 각 모듈은 고유한 `id`를 가져야 합니다
4. **연결 참조**: 연결에서 참조하는 모듈 ID 또는 인덱스가 실제 모듈 배열에 존재해야 합니다
5. **프로젝트 이름**: `name` 또는 `projectName` 필드가 없으면 파일 이름(확장자 제외)이 사용됩니다

## 검증 방법

빌드 후 다음을 확인하세요:
1. `public/samples.json` 파일이 생성되었는지 확인
2. 빌드 로그에서 "Processed X sample files" 메시지 확인
3. 앱의 Samples 메뉴에서 파일이 표시되는지 확인
