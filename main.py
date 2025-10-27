import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram_manual(image):
    # 1. Изчисляване на хистограмата
    # np.bincount е по-бърз от np.histogram за цели числа 
    # image.flatten() превръща 2D масива в 1D
    # minlength=256 гарантира, че имаме 256 bins,
    # дори ако някои нива на сиво липсват в изображението.
    hist = np.bincount(image.flatten(), minlength=256)

    # 2. Изчисляване на кумулативната дистрибутивна функция (CDF)
    # cdf[i] съдържа сумата от hist[0] до hist[i]
    cdf = hist.cumsum()

    # 3. Нормализиране на CDF за създаване на Look-Up Table (LUT)
    
    # Намираме първата не-нулева стойност в CDF.
    # Това е cdf_min според формулата от Wikipedia.
    # Използваме маскиран масив , за да игнорираме нулите.
    # Това е важно за изображения, които нямат пиксели с интензитет 0.
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()

    # Общ брой пиксели = M * N = cdf[-1] (последната стойност в CDF)
    num_pixels = cdf[-1]

    # Формула за трансформация :
    # h(v) = round( (cdf(v) - cdf_min) / (M*N - cdf_min) * (L - 1) )
    # Където L = 256
    L = 256
    
    # Прилагаме формулата за всички стойности от 0 до 255 едновременно
    # (cdf - cdf_min) * (L - 1)
    numerator = (cdf - cdf_min) * (L - 1)
    # (num_pixels - cdf_min)
    denominator = (num_pixels - cdf_min)
    
    # Изчисляваме LUT
    lut = np.round(numerator / denominator)

    # Резултатът трябва да е от тип uint8 
    lut = lut.astype('uint8')

    # 4. Прилагане на LUT към оригиналното изображение
    # NumPy позволява "индексиране" на масив с друг масив .
    # Всеки пиксел със стойност 'v' в 'image' се заменя с 'lut[v]'.
    # Това е изключително бързо и заменя ръчния for-цикъл.
    equalized_image = lut[image]
    
    return equalized_image

image_path = 'dji_fly_20240205_074018_97_1707169913088_photo.jpg'

# 1. Зареждане на изображение
original_image = cv2.imread(image_path)

# Проверка дали изображението е заредено успешно
if original_image is None:
    print(f"ГРЕШКА: Не може да се зареди изображение от '{image_path}'.")
    print("Създаване на синтетично тестово изображение с нисък контраст...")
    # Създаване на просто градиентно изображение с нисък контраст
    original_image = np.fromfunction(lambda i, j: (i + j) // 10 + 80, (400, 500), dtype='uint8')
    is_test_image = True
else:
    is_test_image = False

# 2. Конвертиране в нива на сивото
# Алгоритъмът работи върху едноканални изображения.
if len(original_image.shape) > 2:
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = original_image

# 3. Прилагане на моя алгоритъм
equalized_image = equalize_histogram_manual(gray_image)

# 4. Изчисляване на хистограми за визуализация
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# 5. Визуализация на резултатите с Matplotlib
plt.figure(figsize=(12, 10))
plt.suptitle('Демонстрация на изравняване на хистограма', fontsize=16)

# Входно изображение
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray', vmin=0, vmax=255)
plt.title('Оригинално изображение (Вход)')
plt.axis('off')

# Изходно изображение
plt.subplot(2, 2, 2)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Изравнено изображение (Изход)')
plt.axis('off')

# Хистограма на входа
plt.subplot(2, 2, 3)
plt.plot(hist_original, color='blue')
plt.title('Хистограма на оригинала')
plt.xlabel('Интензитет (0-255)')
plt.ylabel('Брой пиксели')
plt.xlim([0, 256])
plt.grid(True)

# Хистограма на изхода
plt.subplot(2, 2, 4)
plt.plot(hist_equalized, color='red')
plt.title('Хистограма на изравненото')
plt.xlabel('Интензитет (0-255)')
plt.ylabel('Брой пиксели')
plt.xlim([0, 256])
plt.grid(True)

# Показване на графиката
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Запазване на резултата
output_filename = 'equalized_output.png'
cv2.imwrite(output_filename, equalized_image)
print(f"Изравненото изображение е запазено като '{output_filename}'")

if is_test_image:
    input_filename = 'input_test_image.png'
    cv2.imwrite(input_filename, gray_image)
    print(f"Тестовото входно изображение е запазено като '{input_filename}'")