import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def load_images(image_path, mask_path):
    """
    이미지와 마스크를 로드하는 함수
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 마스크 이진화
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return image, mask


def extract_features(image, mask):
    """
    각 객체의 특징을 추출하는 함수
    """
    # 마스크에서 각각의 객체 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    min_area = 1000  # 최소 면적 설정

    for contour in contours:
        # 객체의 면적이 너무 작거나, 마스크와의 겹침이 적은 경우 무시
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 바운딩 박스 찾기
        x, y, w, h = cv2.boundingRect(contour)

        # 마스크 생성
        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [contour], -1, (255), -1)

        # 마스크와의 겹침 비율 확인
        mask_overlap = cv2.countNonZero(cv2.bitwise_and(mask[y:y + h, x:x + w],
                                                        object_mask[y:y + h, x:x + w]))
        if mask_overlap / area < 0.8:  # 마스크와 80% 이상 겹치지 않으면 무시
            continue

        # 객체 영역 추출
        roi = image[y:y + h, x:x + w]
        mask_roi = object_mask[y:y + h, x:x + w]

        # 객체가 마스크 영역에 있는 경우에만 특징 추출
        if cv2.countNonZero(mask_roi) > 0:
            features.append({
                'bbox': (x, y, w, h)
            })

    return features


def classify_defects(features):
    """
    추출된 특징을 기반으로 불량 여부를 판단하는 함수
    """
    results = []
    for feature in features:
        # 마스크가 있는 객체는 모두 불량으로 판정
        results.append({
            'bbox': feature['bbox'],
            'is_defect': True
        })

    return results


def visualize_results(image, results):
    """
    결과를 시각화하는 함수
    """
    result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for result in results:
        x, y, w, h = result['bbox']
        color = (0, 0, 255) if result['is_defect'] else (0, 255, 0)  # 빨강: 불량, 초록: 양품
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)

    return result_img


def create_result_grid(processed_images, grid_size=(3, 3)):
    """
    처리된 이미지들을 그리드 형태로 배치하는 함수
    """
    rows, cols = grid_size
    cell_height = processed_images[0].shape[0]
    cell_width = processed_images[0].shape[1]

    grid = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)

    for idx, img in enumerate(processed_images[:rows * cols]):
        i, j = idx // cols, idx % cols
        grid[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width] = img

    return grid


def process_image(image_path, mask_path):
    """
    전체 프로세스를 처리하는 메인 함수
    """
    image, mask = load_images(image_path, mask_path)
    features = extract_features(image, mask)
    results = classify_defects(features)
    result_img = visualize_results(image, results)
    return result_img


def main():
    image_dir = "Image"
    mask_dir = "Mask"
    os.makedirs("Results", exist_ok=True)

    # 이미지 파일 리스트 가져오기
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    processed_images = []
    grid_count = 0

    print("이미지 처리 시작...")

    for idx, image_file in enumerate(image_files):
        try:
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)

            result_img = process_image(image_path, mask_path)
            processed_images.append(result_img)

            # 9개 이미지마다 그리드 생성
            if len(processed_images) == 9:
                grid = create_result_grid(processed_images)
                grid_count += 1

                # 결과 저장
                cv2.imwrite(os.path.join("Results", f"result_grid_{grid_count}.png"), grid)

                # 결과 표시
                plt.figure(figsize=(15, 15))
                plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
                plt.title('Detection Results')
                plt.axis('off')
                plt.show()

                processed_images = []

            print(f"처리 완료: {idx + 1}/{len(image_files)} - {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    # 남은 이미지 처리
    if processed_images:
        grid = create_result_grid(processed_images)
        grid_count += 1
        cv2.imwrite(os.path.join("Results", f"result_grid_{grid_count}.png"), grid)

        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        plt.title('Detection Results')
        plt.axis('off')
        plt.show()

    print("처리 완료!")


if __name__ == "__main__":
    main()