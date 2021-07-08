## Glass-Recommendation-and-Virtual-Wearing-System

- 사용자의 얼굴형을 분석한 후 얼굴형에 적절한 안경을 추천하고 추천 안경 및 여러 안경들을 가상 착용시켜 주는 시스템
- 얼굴형 분석을 위해 Machine learning object detection 알고리즘인 CascadeClassifier, detectMultiScale, OpenCV를 이용해 얼굴을 인식
- Dlib를 통해 얼굴 속 68개의 landmark를 추출한 후 이를 이용하여 6개의 얼굴형으로 분류
- 분류한 얼굴형에 적합한 안경을 추천하고 눈의 위치를 나타내는 landmark를 이용하여 가상 착용한 모습 보여줌
- 위 기술들을 하나의 GUI를 통해 구현 

#### Landmark 추출
<img src="https://user-images.githubusercontent.com/58179712/124855170-895bd800-dfe3-11eb-8b40-8de599a58804.png"  width="300">

#### 얼굴형 인식
<img src="https://user-images.githubusercontent.com/58179712/124856115-ee63fd80-dfe4-11eb-972f-c259845ddbcf.png"  width="300">

#### 가상 안경 착용
<img src="https://user-images.githubusercontent.com/58179712/124856127-f328b180-dfe4-11eb-978c-056bbdfa37a0.png"  width="300">

#### GUI 구성
<img src="https://user-images.githubusercontent.com/58179712/124856136-f6bc3880-dfe4-11eb-9e68-6b51733e8f4c.PNG"  width="600">
