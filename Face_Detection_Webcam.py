# Face Detection em Tempo Real via Webcam no Google Colab
# Requer permissão para acessar a webcam

# Instalação de dependências
!pip install opencv-python-headless matplotlib

# Importação de bibliotecas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import output
from IPython.display import display, Javascript
import time

# Carregar classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configurar JavaScript para acesso à webcam
js_code = '''
var video;
var div = document.createElement('div');
document.body.appendChild(div);

async function setupWebcam() {
  video = document.createElement('video');
  video.style.display = 'block';
  div.appendChild(video);
  
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();
  
  return true;
}

function captureFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}
'''

display(Javascript(js_code))
output.eval_js('setupWebcam()')

# Função para processamento de frames
def detect_faces():
  try:
    while True:
      # Capturar frame da webcam
      data_url = output.eval_js('captureFrame()')
      
      # Converter URL para imagem OpenCV
      header, data = data_url.split(',', 1)
      bytes_data = np.frombuffer(bytes(data, 'latin-1'), dtype=np.uint8)
      img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
      
      # Detecção facial
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(
          gray,
          scaleFactor=1.1,
          minNeighbors=5,
          minSize=(30, 30)
      )
      
      # Desenhar retângulos
      for (x, y, w, h) in faces:
          cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
      
      # Converter para RGB e exibir
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      plt.imshow(img_rgb)
      plt.axis('off')
      plt.show()
      
      # Limpar saída para atualização
      display.clear_output(wait=True)
      
      # Delay para controle de FPS
      time.sleep(0.1)

  except KeyboardInterrupt:
    # Encerrar acesso à webcam
    output.eval_js('video.srcObject.getTracks().forEach(track => track.stop())')
    print("Webcam liberada!")

# Iniciar detecção
print("Pressione Ctrl+C para parar a detecção")
detect_faces()
