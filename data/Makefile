# Zenodo record URLs
ZENODO_LDGM = https://zenodo.org/record/8157131/files
ZENODO_GRAPHREML = https://zenodo.org/records/15085817/files
ZENODO_SCORES = https://zenodo.org/records/18102484/files

# Download all data files (~25 GB)
download_all: download_precision download_annotations download_scores download_surrogates download_sumstats

# Recommended for graphREML on European-ancestry data (~2 GB)
download_reml: download_ukbb_precision download_annotations download_surrogates

# Download UK Biobank LDGM precision matrices only (~1.5 GB)
download_ukbb_precision:
	mkdir -p ldgms
	curl -L -o - $(ZENODO_GRAPHREML)/precision_ukbb.tgz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	mv ldgms/*/* ldgms 2>/dev/null || true
	ls -d ldgms/*/ 2>/dev/null | xargs rmdir 2>/dev/null || true
	chmod +x create_rsid_position.sh
	./create_rsid_position.sh > rsid_position.csv
	chmod +x rename_ukbb_files.sh
	./rename_ukbb_files.sh

# Download all LDGM precision matrices (~10 GB)
download_precision:
	mkdir -p ldgms
	curl -L -o - $(ZENODO_LDGM)/AFR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/AMR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/EAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/EUR.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/SAS.precision.tar.gz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_GRAPHREML)/precision_ukbb.tgz | tar -xz -C ldgms
	curl -L -o - $(ZENODO_LDGM)/snplists_GRCh38positions.tar.gz | tar -xz -C ldgms
	mv ldgms/*/* ldgms
	ls -d ldgms/*/ | xargs rmdir
	chmod +x create_rsid_position.sh
	./create_rsid_position.sh > rsid_position.csv
	chmod +x rename_ukbb_files.sh
	./rename_ukbb_files.sh

# Download baselineLD annotations (~400 MB)
download_annotations:
	curl -L -o - $(ZENODO_GRAPHREML)/baselineld.tgz | tar -xz

# Download GWAS summary statistics - 20 UK Biobank traits (~7 GB)
download_sumstats:
	mkdir -p sumstats
	curl -L -o - $(ZENODO_GRAPHREML)/sumstats20traits.tgz | tar -xz -C sumstats

# Download all score statistics for enrichment score test (~6.5 GB)
download_scores:
	mkdir -p scores
	curl -L -o scores/baselineld.EAS.genes.h5 $(ZENODO_SCORES)/baselineld.EAS.genes.h5?download=1
	curl -L -o scores/baselineld.EAS.variants.h5 $(ZENODO_SCORES)/baselineld.EAS.variants.h5?download=1
	curl -L -o scores/baselineld.EUR.genes.h5 $(ZENODO_SCORES)/baselineld.EUR.genes.h5?download=1
	curl -L -o scores/baselineld.EUR.variants.h5 $(ZENODO_SCORES)/baselineld.EUR.variants.h5?download=1

# Download gene-level score statistics only (~10 MB)
download_gene_scores:
	mkdir -p scores
	curl -L -o scores/baselineld.EAS.genes.h5 $(ZENODO_SCORES)/baselineld.EAS.genes.h5?download=1
	curl -L -o scores/baselineld.EUR.genes.h5 $(ZENODO_SCORES)/baselineld.EUR.genes.h5?download=1

# Download gene table and surrogate marker files (~60 MB)
download_surrogates:
	curl -L -o genes.tsv $(ZENODO_SCORES)/genes.tsv?download=1
	mkdir -p surrogates
	curl -L -o surrogates/hm3_surrogates.EAS.h5 $(ZENODO_SCORES)/hm3_surrogates.EAS.h5?download=1
	curl -L -o surrogates/hm3_surrogates.UKBB.h5 $(ZENODO_SCORES)/hm3_surrogates.UKBB.h5?download=1
	curl -L -o surrogates/ukbb_surrogates.h5 $(ZENODO_SCORES)/ukbb_surrogates.h5?download=1

.PHONY: download_all download_reml download_ukbb_precision download_precision download_annotations download_sumstats download_scores download_gene_scores download_surrogates
