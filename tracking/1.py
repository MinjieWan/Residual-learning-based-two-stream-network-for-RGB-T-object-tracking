
from PIL import Image as ImagePIL
from PIL import Image
im = ImagePIL.open('1.jpg')
print(im)
print(type(im))
im = cv2.imread('1.jpg')
image = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
image.save('11.jpg',quality=95,dpi=(100.0,100.0))