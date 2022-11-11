El código relevante se encuentra en los archivos Testing_IC (Preprocesamiento y análisis de los datos) y Train_IC (Entrenamiento de clasificadores). Los archivos .ipynb están diseñados para correr el código por el orden de celdas.
Para ejecutar los .ipynb es importante que se tenga la siguiente estructura de carpetas:
root/
|- *.ipynb (* -> Train_IC O Testing_IC)
|- utils.py
|- windows_test.cv (sólo necesario para Train_IC.ipnyb)
|- train/
	|-subj01
	.
	.
	.
	|-subj30
