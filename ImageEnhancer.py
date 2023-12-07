import os
import subprocess

class ImageEnhancer:
    def __init__(
        self, 
        input_path='inputs', 
        model_name='RealESRGAN_x4plus', 
        output_path='results', 
        denoise_strength=0.5, 
        outscale=4, 
        suffix='out', 
        face_enhance=False, 
        fp32=True, 
        gpu_id=None
    ):
        self.input_path = input_path
        self.model_name = model_name
        self.output_path = output_path
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.suffix = suffix
        self.face_enhance = face_enhance
        self.fp32 = fp32
        self.gpu_id = gpu_id

    def enhance_images(self):
        # Crear el directorio si no existe
        os.makedirs(self.output_path, exist_ok=True)

        # Lista de argumentos para el proceso
        args = [
            'python', '../Vision/ProyectoFinal/Real-ESRGAN/inference_realesrgan.py',
            '-i', self.input_path,
            '-o', self.output_path,
            '-n', self.model_name,
            '-s', str(self.outscale),
            '--suffix', self.suffix,
        ]

        if self.denoise_strength:
            args.extend(['-dn', str(self.denoise_strength)])

        if self.face_enhance:
            args.append('--face_enhance')

        if self.fp32:
            args.append('--fp32')

        if self.gpu_id is not None:
            args.extend(['-g', str(self.gpu_id)])

        # Ejecutar el proceso de mejora de imágenes
        subprocess.run(args, check=True)