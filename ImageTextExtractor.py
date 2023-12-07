import os
import math
import pandas as pd
import keras_ocr
import matplotlib.pyplot as plt

from utils import jaccard_index
#keras_ocr.config.configure()

class ImageTextExtractor:
    def __init__(self, images_paths, save_annotated_images, save_folder, df_true_text=None, batch_size=10):
        self.images_paths = images_paths
        self.batch_size = batch_size
        self.save_annotated_images = save_annotated_images
        self.save_folder = save_folder
        self.pipeline = keras_ocr.pipeline.Pipeline()
        self.df_true_text = df_true_text
        self.data = []
        self.global_index = 0 


    def set_df_true_text(self, df_true_text):
        self.df_true_text = df_true_text

    def get_distance(self, predictions):
        x0, y0 = 0, 0 
        detections = []
        for group in predictions:
            top_left_x, top_left_y = group[1][0]
            bottom_right_x, bottom_right_y = group[1][1]
            center_x, center_y = (top_left_x + bottom_right_x)/2, (top_left_y + bottom_right_y)/2
            distance_from_origin = math.dist([x0,y0], [center_x, center_y])
            distance_y = center_y - y0
            detections.append({
                                'text': group[0],
                                'center_x': center_x,
                                'center_y': center_y,
                                'distance_from_origin': distance_from_origin,
                                'distance_y': distance_y
                            })
        return detections

    def distinguish_rows(self, lst, thresh=15):
        sublists = []
        for i in range(0, len(lst)-1):
            if (lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh):
                if lst[i] not in sublists:
                    sublists.append(lst[i])
                sublists.append(lst[i+1])
            else:
                yield sublists
                sublists = [lst[i+1]]
        yield sublists

    def divide_into_batches(self):
        for i in range(0, len(self.images_paths), self.batch_size):
            yield self.images_paths[i:i + self.batch_size]

    def process_images(self, images_batch):
        images = [keras_ocr.tools.read(image_path) for image_path in images_batch]
        prediction_groups = self.pipeline.recognize(images)

        for index, (image_path, predictions) in enumerate(zip(images_batch, prediction_groups)):
            detections = self.get_distance(predictions)
            rows = list(self.distinguish_rows(detections, thresh=15))
            rows = list(filter(lambda x: x != [], rows))

            ordered_preds = []
            for row in rows:
                row = sorted(row, key=lambda x: x['distance_from_origin'])
                for each in row:
                    ordered_preds.append(each['text'])

            if len(image_path.split(".")) > 1:
                self.data.append({'image': image_path.split(".")[0], 'texts': ordered_preds})
            else:
                self.data.append({'image': image_path, 'texts': ordered_preds})

            if self.save_annotated_images:
                fig, ax = plt.subplots(figsize=(20, 10))
                keras_ocr.tools.drawAnnotations(image=images[self.global_index % self.batch_size], predictions=predictions, ax=ax)
                plt.savefig(os.path.join(self.save_folder, f"annotated_image_{self.global_index}.png"))
                plt.close(fig)

            self.global_index += 1  # Incrementa el índice global

    def process_all_images(self):
        for images_batch in self.divide_into_batches():
            self.process_images(images_batch)
        print("Todas las imágenes procesadas con éxito.")

    def get_df_results(self):
        self.df_results = pd.DataFrame(self.data)
        return self.df_results

    def calculate_jaccard_index(self):
        self.df_results["id"] = pd.to_numeric(self.df_results["image"].apply(lambda x: x.split('/', 1)[1]))
        joined_df = self.df_results.merge(self.df_true_text, on='id', how='left')
        joined_df['jaccard_index'] = joined_df.apply(lambda x: jaccard_index(x['texts'], x['true_text']), axis=1)
        return joined_df[["id", "texts", "true_text", "jaccard_index"]], joined_df['jaccard_index'].mean()