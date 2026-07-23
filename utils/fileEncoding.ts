// utils/fileEncoding.ts
// 업로드된 텍스트(CSV) 파일을 "인코딩 자동 감지"로 읽는다.
//
// 문제: FileReader.readAsText()는 항상 UTF-8로 해석하므로, EUC-KR/CP949(윈도우
//       한글 엑셀이 CSV로 저장할 때의 기본 인코딩)로 저장된 파일의 한글 헤더·값이
//       깨져서(���) 보인다. 데이터 로드 시 "한글 제목이 깨져 보이는" 원인.
// 해결: 파일을 ArrayBuffer로 읽어 바이트를 검사한 뒤
//       ① UTF-8 BOM 또는 유효한 UTF-8이면 → UTF-8로 디코드
//       ② 그렇지 않으면 → EUC-KR(=Windows-949/CP949)로 디코드
//       모두 브라우저 내장 TextDecoder만 사용(외부 의존성 없음)하며 결정적이다.
//
// Pyodide 실행/내보낸 Python(pd.read_csv)과는 무관하다. 여기서 얻은 문자열은
// 기존 파서로 그대로 흘러가므로 입력층(파일 읽기)만 보정하는 가산적 수정이다.

/**
 * 바이트 버퍼를 인코딩 자동 감지로 문자열로 디코드한다.
 * UTF-8(BOM/유효) 우선, 실패 시 EUC-KR(CP949) 폴백.
 */
export function decodeTextSmart(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);

  // 1) UTF-8 BOM (EF BB BF) → 명백한 UTF-8. BOM 제거 후 디코드.
  if (
    bytes.length >= 3 &&
    bytes[0] === 0xef &&
    bytes[1] === 0xbb &&
    bytes[2] === 0xbf
  ) {
    return new TextDecoder("utf-8").decode(bytes.subarray(3));
  }

  // 2) 엄격(fatal) 모드로 UTF-8 디코드 시도.
  //    - 순수 ASCII 또는 올바른 UTF-8이면 성공 → UTF-8로 확정.
  //    - EUC-KR 등 비(非)UTF-8 바이트열이면 예외 → 폴백으로 이동.
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    // 3) UTF-8이 아님 → 한글 레거시 인코딩(EUC-KR/CP949)으로 디코드.
    try {
      return new TextDecoder("euc-kr").decode(bytes);
    } catch {
      // 4) 최후의 안전망: 관용 UTF-8(깨진 바이트는 대체문자로).
      //    최소한 앱이 죽지 않고 진행되도록 한다.
      return new TextDecoder("utf-8").decode(bytes);
    }
  }
}

/**
 * 업로드된 File을 인코딩 자동 감지로 읽어 문자열로 반환한다.
 * readAsText(UTF-8 고정) 대신 이 함수를 쓰면 EUC-KR/CP949 한글 CSV도 정상 표시된다.
 */
export function readTextFileSmart(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const buf = e.target?.result;
      if (buf instanceof ArrayBuffer) {
        resolve(decodeTextSmart(buf));
      } else if (typeof buf === "string") {
        resolve(buf);
      } else {
        resolve("");
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}
