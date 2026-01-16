# Samples 폴더

이 폴더에 저장된 파일들이 Samples 메뉴에 자동으로 표시됩니다.

## 지원 파일 형식

### 1. .ins 파일 (Save 버튼으로 저장한 파일)
Save 버튼으로 저장한 `.ins` 파일을 이 폴더에 넣으면 자동으로 표시됩니다.

```json
{
  "modules": [...],
  "connections": [...],
  "name": "My Model"
}
```

### 2. .json 파일 (Samples 형식)
기존 Samples 형식의 JSON 파일도 지원됩니다.

```json
{
  "name": "Sample Model Name",
  "modules": [
    {
      "type": "LoadData",
      "position": { "x": 100, "y": 100 },
      "name": "Load Data",
      "parameters": {
        "source": "data.csv"
      }
    }
  ],
  "connections": [
    {
      "fromModuleIndex": 0,
      "fromPort": "data_out",
      "toModuleIndex": 1,
      "toPort": "data_in"
    }
  ]
}
```

## 사용 방법

1. **Save 버튼으로 모델 저장**:
   - 상단의 "Save" 버튼을 클릭하여 모델을 `.ins` 파일로 저장합니다.
   - 다운로드된 `.ins` 파일을 이 `samples` 폴더에 복사/이동합니다.

2. **Samples 메뉴에서 확인**:
   - Samples 메뉴를 열면 `samples` 폴더의 모든 파일이 자동으로 표시됩니다.
   - 파일을 클릭하면 해당 모델이 로드됩니다.

## 주의사항

- `.json` 또는 `.ins` 확장자를 가진 파일만 읽습니다.
- 파일 이름은 모델 이름으로 표시됩니다 (.ins 파일의 경우 name 또는 projectName 사용).
- 빌드 시점에 `samples` 폴더의 파일들이 `public/samples.json`으로 변환되어 배포됩니다.

