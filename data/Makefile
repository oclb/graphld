download_precision:
	curl -L -o - https://zenodo.org/record/8157131/files/AFR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/AMR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/SAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/records/15085817/files/precision_ukbb.tgz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir
	chmod +x create_rsid_position.sh
	./create_rsid_position.sh > rsid_position.csv
	
download:
	curl -L -o - https://zenodo.org/record/8157131/files/AFR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/AMR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/SAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/records/15085817/files/precision_ukbb.tgz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/records/15085817/files/baselineld.tgz | tar -xz
	curl -L -o scores20traits.baselineld.h5 https://zenodo.org/records/15085817/files/scores20traits.baselineld.h5
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir
	chmod +x create_rsid_position.sh
	./create_rsid_position.sh > rsid_position.csv
	

download_sumstats:
	curl -L -o - https://zenodo.org/records/15085817/files/sumstats20traits.tgz | tar -xz


download_scores:
	curl -L -o scores20traits.baselineld.h5 https://zenodo.org/records/15085817/files/scores20traits.baselineld.h5