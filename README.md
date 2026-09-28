https://github.com/MinY96/ImageProcessingAPI.git
페이지에 있는 최신본을 zip파일로 받은거야.
해당 프로젝트 파일을 기준으로 다음과 같은 작업을 수행해줘

1. 프론트 빌드 오류 수정
2. 평가 Job Manager를 앱 수명주기와 API에 연결
3. 취소, 상태 필터 동작 검증
4. Synthetic NG 탭의 UI/UX 검토
5. 하기 작성한 추가 기능 반영
** 추가 기능 리스트
[backend]
- 이미지 누끼 따기(투명한 배경의 이미지 생성)

[frontend]
- ROI 설정 시 마우스로 영역 클릭하는 기능
[이미지 실험실] UI, UX 수정 필요
	1) Image Viewer | 설정 | Image Viewer 구조
	2) 설정에는
	- Operation, Quick Run 선택 기능
	- Run 기능 존재
	3) 왼쪽/오른쪽 Image Viewer 하단에는 탭으로 Histogram/Image Statistics/Image Feature/Image Analysis 존재
	4) 오른쪽 Image Viewer 우측 하단에 Apply 버튼 생성
	- 버튼 클릭 시 적용된 이미지가 좌측으로 넘어가고
	- 사이드에 History 추가
	- History는 Hide 될 수 있는 패널로 생성
- Image Viewer가 사용되는 모든 컨트롤 우측 상단에 저장 버튼 생성하여 클릭 시 폴더 선택창 떠서 원하는 위치에 저장 가능하도록 기능 추가
)
