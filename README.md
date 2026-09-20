## Easy Embedding V 0.4.2 ##

#### Easy Embedding — ComfyUI용 텍스트 베이스 노드팩####
Stable Diffusion 모델을 재이용하기 쉽도록 하는 도구를 최저한의 요구사항으로
ComfyUI Standalone 환경에서 100% 안정적으로 동작하는 텍스트 처리 노드만 제공합니다.



####Easy Embedding은 다음 원칙을 기반으로 설계되었습니다####

• 환경 안정성 최우선

• 결정성 100% (Portable 환경에서도 동일 결과)

• 유지보수 가능한 구조

이 노드 중에는 comfyui의 기본 노드를 참고해서 만들어진 부분이 있습니다.
ComfyUI의 텍스트 프롬프트, 샘플러 및 임베딩 노드를 연결하기 위한 로직들을 
기반으로 수정된 버전입니다.
# 원본 프로젝트: https://github.com/comfyanonymous/ComfyUI
본 프로젝트는 ComfyUI(AGPL-3.0)를 기반으로 수정되었으며, 원본 라이선스는 external_licenses 디렉토리에 포함되어 있습니다

###requirements.txt가 비어 있는 이유###
ComfyUI는 확장팩 로딩 시 requirements.txt를 자동 설치합니다.
그러나 pip나 git을 통한 원격설치 방식은 환경 손상 위험이 매우 높기 때문에,
requirements.txt를 빈 파일로 유지합니다.

참고로, 포터블의 경우 필수 설치 요소는 다음의 목록뿐입니다. 
에러 안 뜰 경우 설치할 필요 없습니다.

python -m pip install --no-deps torchvision
python -m pip install --no-deps deep-translator beautifulsoup4

## Easy Embedding V 0.0.7 업데이트 정보##
gc(가비지 코드)를 통해 샘플링 노드는 다음단계로 진행할 때마다 실시간으로 진행물 정보를 지웁니다. 
좀 더 저사양에서도 돌아가게 바뀌었습니다.

## Easy Embedding V 0.2.0 업데이트 정보##
이지 샘플러 코드를 조금 더 수정했습니다.
추가 개선의 결과 디퓨저 타입 모델의 사용시에도 오류율이 감소했습니다.

텍스트프롬프트 관련 개선이 있습니다.
텍스트 프롬프트 노드 2종이 추가됩니다.
글로벌 프롬프트 설정이 사라지고 긍정-부정 프롬프트를 기입하는 기능과 텍스트임베딩을 기록하는 기능이 있는 노드와
구글 번역 호출 기능이 있는 번역기 포함 노드입니다. 해당 노드를 지원하기 위해 인스톨해야 하는 항목이 추가됩니다.


## Easy Embedding V 0.2.1 업데이트 정보##
텍스트프롬프트 관련 개선이 있습니다.
쓸모없어진 텍스트 인코더 하나를 정리했습니다.

## Easy Embedding V 0.2.2 업데이트 정보##

샘플러계열들의 처리시 기동부터 모델 로드전까지의 처리상황이 CMD에 로그로 남습니다.
인페인팅용 샘플러 하나를 추가했습니다. 
텍스트프롬프트 관련 개선이 있습니다.
쓸모없어진 텍스트 인코더 두개를 정리했습니다.
@artist 태그와 < : >콜 스타일, () 다중문법의 인식률 개선을 시도하는 중입니다.
인코더+번역기 노드의 경우 번역 진행상황을 알리도록 수정했습니다.
로그체크용 특수 노드 하나가 더 추가되어있습니다. 관심이 있다면 사용해보세요.

## Easy Embedding V 0.4.0 업데이트 정보##
알림 메시지의 표시형식을 바꾸었습니다.
마스크 노드 및 특수기능이 있는 샘플러가 추가됩니다.
테스트 노드들이 몇개 있습니다.
인코더 체커가 이젠 클립비전 지원 여부도 확인됩니다.
인코더+번역기 노드는 무의미하다 판단해 지웠고, 
구글 번역기+커스텀사전 로더로 바꾸었습니다.

## Easy Embedding V 0.4.1 업데이트 정보##
말풍선 관련 노드를 업데이트했습니다.
벡터 리사이징 계열 노드들의 앨리어싱오류를 고쳤고, 이 노드들이 손실나는 구간을 설명에 남겼습니다.

## Easy Embedding V 0.4.2 업데이트 정보##
샘플러 노드의 가비지 콜렉트실행조건을 스위치로 바꿨습니다.
체커 노드들을 4종 추가하고, 미리보기는 텍스트위젯으로 바꿨습니다.
인페인트 보조 노드들이 대량 추가되어있습니다.


#[설치 방법]#



Installation:

1. ZIP 다운로드: 저장소를 .zip 파일로 다운로드합니다.

2. custom_nodes 폴더에 배치: 압축을 풀고 ComfyUI/custom_nodes 폴더에 넣습니다.

3. ComfyUI 재시작: 재시작하면 노드가 로드됩니다.


#[Node Usage Manual]#

[커스텀임베딩 ]

EasyTranslate

- node_id: EasyTranslate
- display_name:번역지원노드

- category: 커스텀임베딩

- 역할: 구글 번역 지원 및 커스텀 번역파일을 로드해 읽는 노드. 
       불리언 스위치를 통해 구글번역을 활용할지, 커스텀 유저 딕셔너리를 쓸지 고를 수 있습니다.

- Inputs: 커스텀 딕셔너리 스위치. 끌 경우 구글번역으로 진행합니다.

- Inputs: 사전파일 선택. 노드 내 translatetext폴더에 저장된 걸 가져옵니다. 기본값으로 예시용 단어 하나가 번역된 파일이 들어있습니다.

- Inputs: 저장 파일명. 커스텀 딕셔너리 사용시엔 적용되지 않습니다.

- Inputs: 번역할 텍스트. 구글 번역이면 문장이어도 되지만, 커스텀 딕셔너리인 경우 단어만으로 번역하는게 나을수도 있습니다.

- Inputs: 타겟 언어

- Outputs: 텍스트 출력


EasyProjLayerLoad
- node_id: EasyProjLayerLoad

- display_name:프로젝션 레이어 로더

- category: 커스텀임베딩/임베드

- 역할: 저장된 프로젝션 레이어를 불러옵니다.

- Inputs: MODEL\proj_embeddings폴더에 있는 프로젝션 레이어 선택지.

- Outputs: 프로젝션 레이어 전달

- Outputs: 저장된 입력 차원(프로젝션 레이어 연결가능)

- Outputs: 저장된 출력 차원(프로젝션 레이어 연결가능)

- Outputs: 저장된 모델 유형(프로젝션 레이어 연결가능)


EasyProjectionLayer
#부정 임베딩용 약화 슬라이더 한계 50.
- node_id: EasyProjectionLayer

- display_name:프로젝션 레이어

- category: 커스텀임베딩/임베드

- 역할: 임베딩 차원을 조건 기대값에 맞게 선형 변환하고 처리합니다.

- Inputs: 임베딩 입력

- Inputs: 프로젝션 레이어 입력

- Inputs: 출력 보정용 클립 연결

- Inputs: 원본 차원 기입

- Inputs: 출력 차원 기입(변경 수치)

- Inputs: 모델 유형 지정

- Inputs: 레이어 저장 여부

- Inputs: 출력 방식 지정

- Inputs: 임베딩 적용 강도

- Inputs: 부정임베딩용 강도 약화 슬라이더. 1로 두면 처리되지 않습니다.

- Outputs: 프로젝션 레이어 저장(연결없음, MODEL\proj_embeddings폴더에 저장)

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력


EasyEmbeddingLoader
- node_id: EasyEmbeddingLoader

- display_name:임베딩 로더

- category: 커스텀임베딩/임베드

- 역할: embeddings 폴더에서 지정한 임베딩(.pt/.safetensors)을 불러와 Projection 출력으로 전달합니다.

- Inputs: MODEL\embeddings폴더에 있는 임베딩 파일 선택지.

- Inputs: 모델 유형 지정

- Inputs: 실행 장치

- Outputs: 임베딩 출력


EasyClipTextEncodeSimple
#특수 입력 키 인식 범위가 증가합니다. EasyClipTextEncode는 중복되어 삭제했습니다.

- node_id: EasyClipTextEncodeSimple
- display_name:CLIP 텍스트 인코더 (단순형)

- category: 커스텀임베딩/프롬프트

- 역할: 텍스트 처리 및 텍스트 임베딩 제작 지원. 이 노드는 강도키워드를 쓸 수 있습니다.
       임베딩 저장시 이 노드에서 이름 및 메타데이터에 키워드기록까지 할 수 있습니다.

- Inputs: 클립 연결

- Inputs: 긍정 텍스트 프롬프트

- Inputs: 부정 텍스트 프롬프트

- Inputs: 출력 설정

- Inputs: 저장될 임베딩 이름

- Inputs: 메타데이터 텍스트키워드 기록

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력(출력노드이기도 하므로 그대로 폴더에 저장됩니다.)

EasyClipTextEncodeTokenInfo
#토큰을 계산하는 기능이 있습니다. 스위치로 진행합니다.
- node_id: EasyClipTextEncodeTokenInfo
- display_name:CLIP 텍스트 인코더 + 번역지원

- category: 커스텀임베딩/프롬프트

- 역할: 텍스트 처리 및 토큰 체크 기능이 있는 노드.

- Inputs: 클립 연결

- Inputs: 긍정 텍스트 프롬프트

- Inputs: 부정 텍스트 프롬프트

- Inputs: 번역할 대상 프롬프트

- Inputs: 번역할 언어

- Inputs: 출력 설정

- Inputs: 저장될 임베딩 이름

- Inputs: 메타데이터 텍스트키워드 기록

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력(출력노드이기도 하므로 그대로 폴더에 저장됩니다.)

- Outputs: 번역된 텍스트 출력


EasyClipTextEncodeADV
#특수 입력 키 인식 범위가 증가합니다

- node_id: EasyClipTextEncodeAdv
- display_name:CLIP 텍스트 인코더 어드밴스

- category: 커스텀임베딩/프롬프트

- 역할: 텍스트 처리 및 텍스트 임베딩 제작 지원. 글로벌 프롬프트가 있어 항목에 맞게 넣고 사용할 수 있습니다.
       이 노드는 강도키워드를 쓸 수 있습니다.
       임베딩 저장시 이 노드에서 이름 및 메타데이터에 키워드기록까지 할 수 있습니다.

- Inputs: 클립 연결

- Inputs: 긍정 텍스트 프롬프트

- Inputs: 부정 텍스트 프롬프트

- Inputs: 출력 설정

- Inputs: 저장될 임베딩 이름

- Inputs: 메타데이터 텍스트키워드 기록

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력(출력노드이기도 하므로 그대로 폴더에 저장됩니다.)


EasyClipTextMask
#복원. 마스크 영역이 처리되는 노드입니다. 훅 샘플러와 같이 쓰는걸 권장합니다.

- node_id: EasyClipTextMask
- display_name:CLIP 텍스트 마스크

- category: 커스텀임베딩/프롬프트

- 역할: 텍스트 처리 및 마스크 영역 프롬프팅 지원.
       이 노드는 강도키워드를 쓸 수 있습니다. 키워드 병합은 마스크를 연결 시엔 적용할 수 없습니다.

- Inputs: 클립 연결

- Inputs: 긍정 텍스트 프롬프트

- Inputs: 부정 텍스트 프롬프트

- Inputs: 출력 설정

- Inputs: 저장될 임베딩 이름

- Inputs: 메타데이터 텍스트키워드 기록

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력(출력노드이기도 하므로 그대로 폴더에 저장됩니다.)


EasyClipTextMask_And_latent
#제작중
- node_id: EasyClipTextMask_And_latent
- display_name:CLIP 텍스트 마스크(참고라텐트 적용)

- category:  커스텀임베딩/프롬프트

- 역할: 텍스트 처리 및 마스크 영역 프롬프팅 지원, 레퍼런스 라텐트 지원예정
       이 노드는 강도키워드를 쓸 수 있습니다. 키워드 병합은 마스크를 연결 시엔 적용할 수 없습니다.

- Inputs: 클립 연결

- Inputs: 긍정 텍스트 프롬프트

- Inputs: 부정 텍스트 프롬프트

- Inputs: 출력 설정

- Inputs: 저장될 임베딩 이름

- Inputs: 메타데이터 텍스트키워드 기록

- Outputs: 컨디셔닝 출력

- Outputs: 임베딩 출력(출력노드이기도 하므로 그대로 폴더에 저장됩니다.)






EasySampler
#기능개선. diffuser model에서도 큰 문제는 없습니다.
 
- node_id: EasySampler

- display_name:간단 샘플러

- category: 커스텀임베딩/샘플링

- 역할: 어드밴스드 샘플러 개량버전. CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.
       시드는 CMD에 표기됩니다. 시드고정을 하고 싶다면 CMD에서 시드를 복사해서 붙이세요.
	   기본적으로 디폴트 시그마 처리가 있어서 flow 모델 계열도 디노이즈 처리가 됩니다.

- Inputs: 사용할 모델

- Inputs: 노이즈 추가 적용여부 확인

- Inputs: 노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다.

- Inputs: 스텝 수

- Inputs: CFG 스케일

- Inputs: 샘플러 알고리즘

- Inputs: 스케줄러

- Inputs: 포지티브 컨디셔닝

- Inputs: 네거티브 컨디셔닝

- Inputs: 라텐트 입력. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다.

- Inputs: 라텐트 에러날 시 대응용 높이조절

- Inputs: 라텐트 에러날 시 대응용 너비조절

- Inputs: 실행 장치

- Inputs: 스텝 시작점

- Inputs: 스텝 종점

- Inputs: 노이즈 반환 처리

- Inputs: 디노이즈 강도

- Outputs: 라텐트 샘플


간단 훅 샘플러
#신규노드. 인페인팅용으로 제작중입니다.
 
- node_id: 간단 훅 샘플러

- display_name:간단 샘플러 어드밴스

- category: 커스텀임베딩/샘플링

- 역할: 이지 샘플러를 훅 지원형태로 개조한 버전. CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.
       시드는 CMD에 표기됩니다. 시드고정을 하고 싶다면 CMD에서 시드를 복사해서 붙이세요.
	   마스크는 마스크 컨디셔닝 타입 노드들로 담아 보내면 알아서 뽑아 처리합니다. 
	   aabb오류나던거 로직 뜯어고쳐서 되게 만들었습니다.

- Inputs: 사용할 모델

- Inputs: 노이즈 추가 적용여부 확인

- Inputs: 노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다.

- Inputs: 스텝 수

- Inputs: CFG 스케일

- Inputs: 샘플러 알고리즘

- Inputs: 스케줄러

- Inputs: 포지티브 컨디셔닝

- Inputs: 네거티브 컨디셔닝

- Inputs: 시그마. 연결 안하면 디폴트 시그마로 돌아갑니다.

- Inputs: 라텐트 입력. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다.

- Inputs: 라텐트 에러날 시 대응용 높이조절

- Inputs: 라텐트 에러날 시 대응용 너비조절

- Inputs: 실행 장치

- Inputs: 노이즈 반환 처리

- Inputs: 디노이즈 강도

- Outputs: 라텐트 샘플




EasySamplerADV
#시그마 적용 가능하게 바꿨습니다. 
 
- node_id: EasySamplerADV

- display_name:간단 샘플러 어드밴스

- category: 커스텀임베딩/샘플링

- 역할: 어드밴스드 샘플러 개량버전. CPU 모드에서도 충돌을 최소화하며 latent를 초기화하고 샘플링을 수행합니다.
       시드는 CMD에 표기됩니다. 시드고정을 하고 싶다면 CMD에서 시드를 복사해서 붙이세요.

- Inputs: 사용할 모델

- Inputs: 노이즈 추가 적용여부 확인

- Inputs: 노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다.

- Inputs: 스텝 수

- Inputs: CFG 스케일

- Inputs: 샘플러 알고리즘

- Inputs: 스케줄러

- Inputs: 포지티브 컨디셔닝

- Inputs: 네거티브 컨디셔닝

- Inputs: 시그마. 연결 안하면 디폴트 시그마로 돌아갑니다.

- Inputs: 마스크 입력. 영역 인페인팅을 시도할 수 있습니다.

- Inputs: 라텐트 입력. 빈 라텐트 이미지를 넣거나, 인코딩 또는 불러온 라텐트를 연결할 수 있습니다.

- Inputs: 라텐트 에러날 시 대응용 높이조절

- Inputs: 라텐트 에러날 시 대응용 너비조절

- Inputs: 실행 장치


- Inputs: 노이즈 반환 처리

- Inputs: 디노이즈 강도

- Outputs: 라텐트 샘플


EasyEmbedLoader
- node_id: EasyEmbedLoader

- display_name:BIN PT 임베딩 로더

- category: 커스텀임베딩/임베딩

- 역할: .bin 또는 .pt 포맷 임베딩을 불러옵니다. 필요시 변환 노드에서 다른 포맷으로 저장하세요. 이 노드는 레이어 연결용이 아닙니다. 

- Inputs: MODEL\embeddings폴더에 있는 .bin 또는 .pt 포맷 임베딩 파일 선택지.

- Inputs: 파일명 출력 여부

- Outputs: 임베딩 객체

- Outputs: 불러온 파일 이름 (옵션, 변환 노드에 연결 가능합니다.)




EasyEmbedTransform
- node_id: EasyEmbedTransform

- display_name:BIN/PT → EMBEDDING 변환

- category: 커스텀임베딩/임베딩

- 역할: .bin 또는 .pt 포맷 임베딩을 safetensors로 변환하여 proj_embeddings에 저장합니다. 용량이 너무 적으면 에러방지용 더미텐서가 출력된겁니다.

- Inputs: 연결된 BIN/PT 포맷 임베딩 객체

- Inputs: 저장할 파일 이름 (옵션)

- Outputs: image


EasyencoderChecker
#성능 업. 이제 노드위젯으로 직접 텍스트를 확인합니다.
- node_id: EasyencoderChecker

- display_name:CLIP 인코더 체크(시각화)

- category: 커스텀임베딩/특수

- 역할: CLIP 인코더의 타입과 차원을 텍스트로 시각화합니다.

- Inputs: 확인할 텍스트 인코더 파일

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: text

EasyEmbeddingChecker
#성능 업. 이제 노드위젯으로 직접 텍스트를 확인합니다.
- node_id: EasyEmbeddingChecker

- display_name:임베딩 체크(시각화)

- category: 커스텀임베딩/특수

- 역할: 임베딩 파일 구조와 shape을 텍스트로 시각화합니다.

- Inputs: 확인할 임베딩 파일

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: text




EasyProjLayerChecker
#성능 업. 이제 노드위젯으로 직접 텍스트를 확인합니다.
- node_id: EasyProjLayerChecker

- display_name:프로젝션 레이어 체크(시각화)

- category: 커스텀임베딩/특수

- 역할: 프로젝션 레이어 파일의 차원/메타데이터를 확인하고 오류 여부를 시각화합니다.

- Inputs: 확인할 프로젝션 레이어 파일

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: text


EasyLoraVersionChecker
#성능 업. 이제 노드위젯으로 직접 텍스트를 확인합니다.
- node_id: EasyLoraVersionChecker

- display_name:LoRA 버전 체커

- category: 커스텀임베딩/특수

- 역할: 로라/리코리스의 정보를 텍스트로 시각화합니다.

- Inputs: 확인할 로라/리코리스 파일

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Inputs: 표시할 키워드 수. 상위 20개까지 추출합니다.

- Inputs: 텍스트 파일로 저장 여부

- Outputs: 텍스트 



easyEmbeddingToTextNode
- node_id: easyEmbeddingToTextNode

- display_name:임베딩 → 텍스트 변환

- category: 커스텀임베딩/특수

- 역할: 메타데이터에 키워드가 등록이 된 임베딩에서 키워드를 가져옵니다.
       저장된 safetensors 임베딩 파일의 메타데이터에서 원본 텍스트를 복원합니다.

- Inputs: 확인할 임베딩 파일

- Inputs: 저장 여부

- Outputs: 텍스트 딕셔너리. comfyui의 기본 노드인 show any에 연결할 수 있습니다.


easyTextLoader

- node_id: easyTextLoader

- display_name:텍스트 로더

- category: 커스텀임베딩/특수

- 역할: 저장된 텍스트 파일의 키워드를 텍스트 프롬프트에 전달합니다.

- Inputs: 확인할 텍스트 파일

- Outputs: 텍스트(String). comfyui의 기본 노드인 clip text prompt의 텍스트란에 연결할 수 있습니다.



EasyTextslot_Loader
#신규노드
- node_id: EasyTextslot_Loader

- display_name:텍스트 슬롯 로더

- category: 커스텀임베딩/특수

- 역할: 이미지에 레퍼런스 텍스트 박스를 넣습니다.

- Inputs: 입력할 레퍼런스 이미지. 노드 내에 있는 bubble_layout폴더에 추가 레퍼런스 이미지를 넣어 쓸수도 있습니다.

- Inputs: 텍스트 박스 라인, 외곽선, 박스 외곽 색상을 정합니다.

- Inputs: 텍스트 슬롯 크기, 텍스트슬롯 좌표, 텍스트 설정, 색상값 지정, 투명도지정

- Outputs: 레이아웃 정보 전달

EasyTextslot
#기능개선
- node_id: EasyTextslot

- display_name:텍스트 이미지 기입

- category: 커스텀임베딩/특수

- 역할: 이미지에 텍스트 키를 넣습니다.

- Inputs: 확인할 폰트 파일(첫 실행시 모델 폴더에 fonts 폴더가 생성, 여기에 폰트 파일[*.ttf, *.ttc]들을 넣으면 됩니다.)

- Inputs: 폰트 크기

- Inputs: 레이아웃 정보(옵션)

- Inputs: 텍스트 슬롯 크기, 텍스트슬롯 좌표, 텍스트 설정, 색상값 지정, 투명도지정

- Inputs: 입력할 텍스트 칸. 입력한 대로 텍스트 슬롯에 나옵니다.

- Inputs: 원본 이미지

- Outputs: image

LogTranslate

- node_id: LogTranslate

- display_name:로그 번역기

- category: 커스텀임베딩/특수

- 역할: ComfyUI 에러 로그를 줄바꿈/필터링 후 번역하여 에러 지점을 빠르게 확인

- Inputs: 텍스트

- Inputs: 텍스트정렬/번역여부/출력여부

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: 텍스트


EasyUnetChecker
#신규노드
- node_id: EasyUnetChecker

- display_name:UNet/DiT 모델 분석기

- category: 커스텀임베딩/특수

- 역할: 선택된 UNet 또는 DiT 모델 파일을 분석하여 파라미터 수, VRAM 추정치, 데이터타입 등을 시각화합니다.

- Inputs: 확인할 UNet 또는 DiT 모델

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: 텍스트

EasyVAEChecker
#신규노드
- node_id: EasyVAEChecker

- display_name:VAE 체크 및 인코딩 테스트

- category: 커스텀임베딩/특수

- 역할: 선택한 VAE의 구조 분석, VRAM 추정 및 실제 Encode/Decode 테스트 결과를 시각화합니다..

- Inputs: 확인할 VAE 모델

- Inputs: 폰트 크기

- Inputs: 미리보기 확인창. 체크시 노드에서 바로 텍스트를 봅니다.

- Outputs: 텍스트


EasyControlNetChecker
#신규노드
- node_id: EasyControlNetChecker

- display_name:ControlNet 모델 분석기

- category: 커스텀임베딩/특수

- 역할: 선택된 ControlNet 모델 파일을 분석하여 파라미터, VRAM 추정치, 데이터타입 등을 시각화합니다.

- Inputs: 확인할 ControlNet 모델

- Inputs: 폰트 크기

- Inputs: 노드 2.0 사용여부. 킬 경우 텍스트 에어리어 위젯으로 열립니다. 끈 경우 일반 텍스트 위젯으로 엽니다.

- Outputs: 텍스트