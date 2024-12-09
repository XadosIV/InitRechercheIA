import torch
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

class CatToyDataset(torch.utils.data.Dataset):
	def __init__(self, images, transform=None):
		self.images = images
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		if self.transform:
			image = self.transform(image)

		return image


def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()


if __name__ == '__main__':
	#Charger les données
	dataset = load_dataset("diffusers/cat_toy_example", download_mode="reuse_cache_if_exists")
	transform = transforms.Compose([
		transforms.Resize((512,512)),
		transforms.ToTensor()
	])
	train_images = dataset["train"]["image"]
	train_dataset = CatToyDataset(train_images, transform=transform)
	trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

	#Charger le batch d'images
	dataiter = iter(trainloader)
	images = next(dataiter)

	#Tensor to Pil
	transform_to_pil = transforms.ToPILImage()
	pil_images = [transform_to_pil(img) for img in images]

	pil_images[0].save(f"outputstable/catSource_0.png")
	pil_images[1].save(f"outputstable/catSource_1.png")


	pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
	pipe = pipe.to("cuda")

	# Prompt pour guider la génération
	prompt = "A cute toy cat, fantasy style"
	img_to_img = pipe(
        prompt=[prompt] * len(pil_images),
        image=pil_images,
        strength=0.5,  # Contrôle la déformation 
        			   # (0.0 = pas de changement, 1.0 = complètement altéré)
        guidance_scale=8.5  # Contrôle le guidage textuel
    ).images

	for i,img in enumerate(img_to_img):
		img.save(f"outputstable/catToy_{i}.png")




	# Charger le modèle Stable Diffusion (Text-to-Image)
	#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
	#pipe = pipe.to("cuda")

	# Text-to-Image : Génération à partir d'un texte
	#prompt = "A cute toy cat sitting on a chair, realistic style"
	#text_to_image = pipe(prompt=prompt, guidance_scale=7.5).images[0]
	#text_to_image.show()
	
	# Afficher un batch d'images originales
	#dataiter = iter(trainloader)
	#images = next(dataiter)
	#imshow(torchvision.utils.make_grid(images))

	# Sauvegarder les images générées
	#text_to_image.save("output/text_to_image_example.png")

