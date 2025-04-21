url = https://drive.usercontent.google.com/u/0/uc?id=11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E&export=download

hw01:
	python -B src/K_mean_cluster.py

hw02:
	python -B src/kmean_pca.py
	python -B src/scipy_linkage_blobs.py
	python -B src/scipy_linkage_moons.py

syn1:
	python -B src/syn1.py

data:
	mkdir data
	cd data; curl -LO $(url);
	unzip MNIST_ORG.zip

clean:
	rm data/*

syn2_prep:
	python -B src/generate_mnist_csv.py

syn2:
	python -B src/syn_2.py