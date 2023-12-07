import os
import re
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from utils import jaccard_index

class ImageObjectDetectionYOLO:
    def __init__(self, model_name, confidence, predict_folder, df_true_objects, batch_size=10, absolute_path = False) -> None:
        if absolute_path:
            self.model = YOLO(model_name)
        else:   
            self.model = YOLO('content/'+model_name+'.pt')
        self.predict_folder = predict_folder
        self.confidence = confidence
        self.df_true_objects = df_true_objects
        self.batch_size = batch_size
        self.PIL_images = self.load_images()
        self.results = []
    
    def load_images(self):
        images = []
        for file in os.listdir(self.predict_folder):
            if file.endswith(".jpg"):
                images.append(Image.open(self.predict_folder+'/'+file))
        return images
    
    def divide_into_batches(self):
        for i in range(0, len(self.PIL_images), self.batch_size):
            yield self.PIL_images[i:i + self.batch_size]

    def predict(self):
        all_results = []
        for batch in self.divide_into_batches():
            results = self.model.predict(source=batch, conf=self.confidence, save=True, project=self.predict_folder, exist_ok=True)
            batch_results = {
                "image": [result.path for result in results],
                "predictions": [[result.names[cls.item()].lower() for cls in result.boxes.cls] for result in results]
            }
            all_results.append(batch_results)
        self.df_results = pd.concat([pd.DataFrame.from_dict(batch) for batch in all_results])
        #self.df_results['predictions'] = self.df_results['predictions'].apply(lambda x: [item.lower() for sublist in x for item in sublist])    

    def get_df_results(self):
        return self.df_results

    def calculate_jaccard_index(self):
        self.df_results["id"] = pd.to_numeric(self.df_results["image"].apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else None))
        joined_df = self.df_results.merge(self.df_true_objects, on='id', how='left')
        joined_df['jaccard_index'] = joined_df.apply(lambda x: jaccard_index(x['predictions'], x['true_objects']), axis=1)
        return joined_df[["id", "predictions", "true_objects", "jaccard_index"]], joined_df['jaccard_index'].mean()



    def train_model(self, train_conf_path, new_model_name, epochs=10, imgsz= 640, ):
        self.config_yaml = os.path.join(train_conf_path, 'config.yaml')
        project = 'xix_print_images'
        project_name = 'training_process'
        results = self.model.train(data= self.config_yaml , imgsz=imgsz, epochs=epochs, project=project, name=project_name, exist_ok=True) 
        #self.model.save(os.path.join('content', new_model_name + '.pt'))
        best_model_path = os.path.join(project, project_name, 'weights', 'best.pt')
        return best_model_path, results
