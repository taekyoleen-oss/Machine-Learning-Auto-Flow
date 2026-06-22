# Azure ML 책 기반 분석·문서 모음

Jeff Barnes, *Microsoft Azure Essentials: Azure Machine Learning* (Microsoft Press, 2015)을 근거로
**ML Auto Flow**(베이스 기준, JMDC 영향 표기)를 분석·문서화한 결과물입니다.

| 문서 | 내용 |
|---|---|
| [01_book_based_improvements.md](01_book_based_improvements.md) | 책 기반 **개선 사항 + 추가 가능 기능**(부스팅 트리·추천·배포·재학습 등, 우선순위/난이도/재현성·동기화 영향 포함) |
| [02_app_booklet_direction.md](02_app_booklet_direction.md) | 책처럼 **앱을 설명하는 책자** 제작 방향·가능성·목차(안)·포맷·작업량 |
| [03_book_models_reproduction.md](03_book_models_reproduction.md) | 책의 **데이터·모델을 앱으로 재현**한 전체 과정 + 실제 검증 결과(회귀/분류/군집) |

## 실제 생성된 실행 가능 산출물 (문서 03)
- 데이터셋: `verify/datasets/imports-85-hdrs.csv`, `adult.csv`, `wholesale_customers.csv` (UCI 공개데이터, 헤더 정리본)
- 검증 픽스처: `verify/pipelines/11_book_automobile_linreg.json`, `12_book_wholesale_kmeans.json`, `13_book_adult_clf.json`
- 앱 샘플: `samples/Book_Automobile_LinearRegression.json`, `Book_AdultIncome_DecisionTree.json`, `Book_Wholesale_KMeans.json`
- 검증: `npm run verify:pipelines` → **12/12 PASS** (외부 Python 2회 byte-identical 재현)

> 본 문서들은 **계획·제안 + 검증된 재현 자료**이며, 산출물 01의 신규 앱 기능 구현은 사용자 검토·승인 후 별도 진행합니다.
