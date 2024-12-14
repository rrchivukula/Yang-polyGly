args <- commandArgs(trailingOnly=TRUE)

path.samples <- args[1]
group.A <- args[2]
group.B <- args[3]
path.counts <- args[4]
path.counts.normalization <- args[5]
path.output <- args[6]

suppressMessages(library(DESeq2))

samples <- read.delim(path.samples, header = T, sep = "\t", check.names = F, comment.char = "#")
samples <- samples[, c("sample", "group")]
samples$group <- relevel(factor(samples$group), ref = group.A)
# print(samples)

counts <- read.delim(path.counts, header = T, sep = "\t", check.names = F, comment.char = "#")
row.names(counts) <- counts[, 1]
counts <- counts[, -1]
# print(head(counts))

counts.normalization <- read.delim(path.counts.normalization, header = T, sep = "\t", check.names = F, comment.char = "#")
row.names(counts.normalization) <- counts.normalization[, 1]
counts.normalization <- counts.normalization[, -1]
# print(head(counts.normalization))

dds.normalization <- DESeqDataSetFromMatrix(countData = counts.normalization, colData = samples, design = ~ group)
dds.normalization <- estimateSizeFactors(dds.normalization)
print(sizeFactors(dds.normalization))

dds <- DESeqDataSetFromMatrix(countData = counts, colData = samples, design = ~ group)
# dds <- estimateSizeFactors(dds)
sizeFactors(dds) <- sizeFactors(dds.normalization)
dds <- estimateDispersions(dds)
dds <- nbinomWaldTest(dds)

plotDispEsts(dds)

res <- results(dds, independentFiltering = FALSE, contrast = c("group", group.B, group.A))
print(head(res))

res$gene <- rownames(res)
res <- res[, c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")]

write.table(as.matrix(res), file = path.output, sep = "\t", row.names = F, quote = F)

counts.normalization <- read.delim(path.counts.normalization, header = T, sep = "\t", check.names = F, comment.char = "#")
row.names(counts.normalization) <- counts.normalization[, 1]
counts.normalization <- counts.normalization[, -1]