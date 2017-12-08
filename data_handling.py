from utils import remove_supersets

class Proofs():
    def __init__(self):
        self.proofs = {}

    def __len__(self):
        return len(self.proofs)

    def __getitem__(self, theorem):
        return self.proofs[theorem]

    def add(self, theorem, proof):
        proof = set(proof)
        if not theorem in self.proofs:
            self.proofs[theorem] = [proof]
        else:
            for prf in self.proofs[theorem]:
                if proof >= prf:
                    break
                if proof < prf:
                    prf &= proof
                    break
            else:
                self.proofs[theorem].append(proof)
            self.proofs[theorem] = remove_supersets(self.proofs[theorem])

    def update(self, new_proofs):
        for thm in new_proofs:
            for prf in new_proofs[thm]:
                self.add(thm, prf)

    def nums_of_proofs(self):
        return [len(self[t]) for t in self.proofs]

    def num_of_all_proofs(self):
        return sum(self.nums_of_proofs())

    def avg_num_of_proofs(self):
        return self.num_of_all_proofs() / len(self)

class Rankings():
    def __init__(self):
        self.rankings = {}

    def __len__(self):
        return len(self.rankings)

    def __getitem__(self, theorem):
        return self.rankings[theorem]

    def add(self, theorem, ranking):
        self.rankings[theorem] = ranking


if __name__ == "__main__":
    prfs = Proofs()
    prfs.add("t1", ["p3", "p2"])
    prfs.add("t1", ["p5", "p2"])
    prfs.add("t2", ["p1"])
    prfs.add("t1", ["p4"])
    prfs.add("t1", ["p2"])
    print(prfs.num_of_all_proofs())
    print(prfs.nums_of_proofs())
    print(prfs["t1"])
