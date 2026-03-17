import pandas as pd
import matplotlib.pyplot as plt


# 나이 값을 받아 연령대 문자열로 변환하는 함수
def get_age_group(age):
    # 결측치인 경우 연령대를 나눌 수 없으므로 Unknown 반환
    if pd.isna(age):
        return 'Unknown'

    # 연령대 구분을 위해 정수형으로 변환
    age = int(age)

    if 10 <= age < 20:
        return '10s'
    if 20 <= age < 30:
        return '20s'
    if 30 <= age < 40:
        return '30s'
    if 40 <= age < 50:
        return '40s'
    if 50 <= age < 60:
        return '50s'
    if 60 <= age < 70:
        return '60s'
    if 70 <= age < 80:
        return '70s'

    #범위 밖의 나이는 Others로 처리
    return 'Others'


# Transported와 가장 관련성이 높은 항목을 찾는 함수
def find_most_related_column(train_df):
    # 기준이 되는 목표 컬럼
    target_column = 'Transported'

    # 식별자 성격이 강해서 분석 의미가 떨어지는 컬럼은 제외함
    excluded_columns = {'PassengerId', 'Name', 'Cabin'}

    # 각 컬럼의 관련성 점수를 저장할 딕셔너리
    scores = {}

    # train_df의 모든 컬럼을 순회
    for column in train_df.columns:
        # 목표 컬럼 자체와 제외 대상 컬럼은 건너뜀
        if column == target_column or column in excluded_columns:
            continue

        # 현재 컬럼과 목표 컬럼만 가져오고 결측치는 제거
        temp_df = train_df[[column, target_column]].dropna()

        # 사용할 데이터가 없으면 건너뜀
        if temp_df.empty:
            continue

        # 불리언 타입 컬럼인 경우
        if pd.api.types.is_bool_dtype(temp_df[column]):
            # 각 값별 Transported 평균을 계산
            grouped = temp_df.groupby(column, observed=False)[target_column].mean()

            # 가장 큰 평균과 가장 작은 평균의 차이를 관련성 점수로 사용
            score = grouped.max() - grouped.min()
            scores[column] = score

        # 숫자형 타입 컬럼인 경우
        elif pd.api.types.is_numeric_dtype(temp_df[column]):
            # 숫자형 컬럼과 Transported(True/False)를 1/0으로 바꾼 값의 상관계수 계산
            correlation = temp_df[column].corr(temp_df[target_column].astype(int))

            # 상관계수가 정상값일 때만 저장
            if pd.notna(correlation):
                # 절대값을 사용하여 관련성의 크기만 비교
                scores[column] = abs(correlation)

        # 문자열 등 범주형 컬럼인 경우
        else:
            # 각 범주별 Transported 평균을 계산
            grouped = temp_df.groupby(column, observed=False)[target_column].mean()

            # 범주 간 평균 차이를 관련성 점수로 사용
            score = grouped.max() - grouped.min()
            scores[column] = score

    # 점수가 하나도 없으면 None 반환
    if not scores:
        return None, None

    # 가장 점수가 높은 컬럼 선택
    best_column = max(scores, key=scores.get)

    # 가장 관련성 높은 컬럼명과 점수를 반환
    return best_column, scores[best_column]


# 연령대별 Transported 여부를 하나의 그래프로 시각화하는 함수
def draw_transport_by_age_group(train_df):
    # 그래프에 표시할 연령대 순서 정의
    age_order = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']

    # Age와 Transported 컬럼만 사용하고 결측치 제거
    graph_df = train_df[['Age', 'Transported']].dropna().copy()

    # 나이 값을 연령대 문자열로 변환하여 새 컬럼 생성
    graph_df['AgeGroup'] = graph_df['Age'].apply(get_age_group)

    # 10대~70대 데이터만 남김
    graph_df = graph_df[graph_df['AgeGroup'].isin(age_order)]

    # 연령대와 Transported 여부별 인원 수를 계산
    result = graph_df.groupby(['AgeGroup', 'Transported'], observed=False).size()

    # 보기 쉽게 행렬 형태로 변환
    result = result.unstack(fill_value=0)

    # 연령대 순서를 10대~70대로 고정
    result = result.reindex(age_order, fill_value=0)

    # 그래프 범례를 보기 쉽게 이름 변경
    result = result.rename(columns={False: 'Not Transported', True: 'Transported'})

    # 혹시 특정 값이 아예 없으면 컬럼을 0으로 생성
    if 'Not Transported' not in result.columns:
        result['Not Transported'] = 0
    if 'Transported' not in result.columns:
        result['Transported'] = 0

    # 두 항목을 막대그래프로 그림
    result[['Not Transported', 'Transported']].plot(kind='bar')

    # 그래프 제목 설정
    plt.title('Transported by Age Group')

    # x축 이름 설정
    plt.xlabel('Age Group')

    # y축 이름 설정
    plt.ylabel('Count')

    # x축 글자 기울기 제거
    plt.xticks(rotation=0)

    # 레이아웃 자동 조정
    plt.tight_layout()



# Destination별 승객 연령대 분포를 시각화하는 함수
def draw_destination_age_distribution(merged_df):
    # 그래프에 사용할 연령대 순서 정의
    age_order = ['10s', '20s', '30s', '40s', '50s', '60s', '70s']

    # Destination과 Age 컬럼만 사용하고 결측치 제거
    bonus_df = merged_df[['Destination', 'Age']].dropna().copy()

    # 나이 값을 연령대 문자열로 변환하여 새 컬럼 생성
    bonus_df['AgeGroup'] = bonus_df['Age'].apply(get_age_group)

    # 10대~70대 데이터만 사용
    bonus_df = bonus_df[bonus_df['AgeGroup'].isin(age_order)]

    # 목적지와 연령대별 인원 수 계산
    result = bonus_df.groupby(['Destination', 'AgeGroup'], observed=False).size()

    # 보기 쉽게 표 형태로 변환
    result = result.unstack(fill_value=0)

    # 열 순서를 10대~70대로 고정
    result = result.reindex(columns=age_order, fill_value=0)

    # 전치해서 연령대가 x축에 오도록 막대그래프 생성
    result.T.plot(kind='bar')

    # 그래프 제목 설정
    plt.title('Age Group Distribution by Destination')

    # x축 이름 설정
    plt.xlabel('Age Group')

    # y축 이름 설정
    plt.ylabel('Count')

    # x축 글자 기울기 제거
    plt.xticks(rotation=0)

    # 레이아웃 자동 조정
    plt.tight_layout()




# 실행하는 메인 함수
def main():
    # train.csv 읽기
    train_df = pd.read_csv('train.csv')

    # test.csv 읽기
    test_df = pd.read_csv('test.csv')

    # 두 데이터를 하나로 병합
    merged_df = pd.concat([train_df, test_df], ignore_index=True)

    # 데이터 개수 출력
    print('=== Data Count ===')
    print(f'train.csv rows: {len(train_df)}')
    print(f'test.csv rows: {len(test_df)}')
    print(f'merged rows: {len(merged_df)}')
    print()

    # train 데이터에서 Transported와 가장 관련성 높은 컬럼 찾기
    best_column, score = find_most_related_column(train_df)

    # 관련성 분석 결과 출력
    print('=== Most Related Column to Transported ===')
    if best_column is not None:
        print(f'column: {best_column}')
        print(f'score: {score}')
    else:
        print('No related column found.')
    print()

    # 연령대별 Transported 여부 그래프 출력
    draw_transport_by_age_group(train_df)

    # Destination별 연령대 분포 그래프 출력
    draw_destination_age_distribution(merged_df)

    # 그래프 출력
    plt.show()

# 현재 파일이 직접 실행될 때만 main() 함수 실행
if __name__ == '__main__':
    main()
