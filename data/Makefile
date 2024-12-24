download:
	mkdir ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/AFR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/AMR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/SAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/sumstats.tar.gz| tar -xz
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir
	
download_eur:
	mkdir ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir
	
download_all:
	mkdir ldgms
	curl -L -o ldgms/1kg_nygc_trios_removed_All_pops_geno_ids_pops.csv https://zenodo.org/record/8157131/files/1kg_nygc_trios_removed_All_pops_geno_ids_pops.csv
	curl -L -o - https://zenodo.org/record/8157131/files/AFR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/AMR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/SAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/sumstats.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/ldgms.tar.gz | tar -xz -C ldgms
	curl -L -o - https://zenodo.org/record/8157131/files/ukb.tar | tar -xz -C ldgms
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir

