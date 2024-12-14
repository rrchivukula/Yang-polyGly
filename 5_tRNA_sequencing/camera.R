args <- commandArgs(trailingOnly=TRUE)

path.samples <- args[1]
group.A <- args[2]
group.B <- args[3]
path.dir <- args[4]

path.counts <- paste(path.dir, "counts.csv", sep = "/")
path.stats <- paste(path.dir, "stats.csv", sep = "/")
path.gene.set <- paste(path.dir, "set.csv", sep = "/")
path.output <- paste(path.dir, "result.csv", sep = "/")

samples <- read.delim(path.samples, header = T, sep = "\t", check.names = F, comment.char = "#")
samples <- samples[, c("sample", "group")]
samples$group <- relevel(factor(samples$group), ref = group.A)
# head(samples)

counts <- read.delim(path.counts, header = T, sep = "\t", check.names = F, comment.char = "#")
row.names(counts) <- counts[, 1]
counts <- counts[, -1]
# head(counts)

stats <- read.delim(path.stats, header = T, sep = "\t", check.names = F, comment.char = "#")
stats <- setNames(stats[, 2], stats[, 1])
# head(stats)

gene.set_ <- read.delim(path.gene.set, header = T, sep = "\t", check.names = F, comment.char = "#")
gene.set <- gene.set_[,2]
gene.set.name <- colnames(gene.set_)[2]
# head(gene.set)

gene.set.list <- list()
gene.set.list[[gene.set.name]] <- gene.set
# print(gene.set.list)

library(limma)
design <- cbind(Intercept=1, Group=samples$group)
cor.est <- interGeneCorrelation(counts, design)
print(cor.est)

result <- cameraPR(stats, gene.set.list, use.ranks=FALSE, inter.gene.cor=cor.est$correlation)
print(result)

write.table(result, file = path.output, sep = "\t", row.names = T, quote = F)