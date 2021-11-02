# Handwritten
Với Deep learning thì bài toán nhận diện chữ viết tay cũng không phải là quá mới mẻ, với data gồm 26 chữ cái trong Tiếng Anh trên [Kaggle](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format) chúng ta hoàn toàn có thể xây dựng mô hình huấn luyện để nhận biết 26 class (tương ứng với 26 chữ cái). Ở trong project này mình sẽ thực hiện dự đoán chữ cái bằng cái viết trực tiếp với thư viện Gradio.

<p align="center">
	<img src="https://github.com/KudoKhang/Hand-Written/blob/main/imgTest/handwritten.gif?raw=true" />
</p>

# How it work
## Phần 1: Huấn luyện mô hình
- Chi tiết huấn luyện từng bước ở [TrainingModel.ipynb](https://github.com/KudoKhang/Hand-Written/blob/main/TrainingModel.ipynb)
## Phần 2: Thực hiện dự đoán
Với [Gradio](https://gradio.app), nó cung cấp cho chúng ta cách để triển khai dự đoán các mô hình máy học một cách nhanh nhất và thuận tiện nhất.  Trong dự án này, với input là một chữ viết tay Gradio cung cấp cho chùng ta một phương thức là Sketchpad, giúp ta có vẽ chữ viết tay lên một cách rất thuận tiện.

<p align="center">
	<img src="https://github.com/KudoKhang/Hand-Written/blob/main/imgTest/3.png?raw=true" />
</p>

Đầu tiên chúng ta cần khai báo model mà chúng ta đã huấn luyện:

```python
from tensorflow.keras.models import load_model
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
model = load_model('modelHandWritten.h5')
```

Xử lý input và dự đoán kết quả:

```python
def classify(img):
	img_final = cv2.resize(img, (28, 28))
	img_final = np.reshape(img_final, (1, 28, 28, 1))
	prediction = model.predict(img_final).flatten()
	return {word_dict[i]: float(prediction[i]) for i in range(25)}
```

Hiển thị lên:

```python
iface = gr.Interface(
	classify,
	gr.inputs.Image(shape=(224, 224), image_mode='L', invert_colors=True, source="canvas"),
	gr.outputs.Label(num_top_classes=3),
	capture_session=True,
	)
```

Nếu muốn nó tạo link để dự đoán online thì ta cần set `share=True`

```python
if __name__ == "__main__":
	iface.launch(share=True)
```

# Usage
Để sử dụng project:
```bash
git clone https://github.com/KudoKhang/Hand-Written
cd Hand-Written
python aoo.py
```
Sau khi đã có thể nhận diện những chữ cái viết tay như vậy chúng ta có thể kết hợp với các kỹ thuật xử lý ảnh để xây dựng nên một ứng dụng như **"Scan & Chấm bài trắc nghiệm"**, **"Dự đoán chữ cái bằng cách dùng bút vẽ trước camera"**...