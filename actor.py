class Actor:
    def __init__(self, *args):
        self.sum = 0
        self.aces = 0
        [self.add_card(x) for x in args]
    
    def add_card(self, card):
        self.aces += 1 if card == 11 else 0
        self.sum += card
        while (self.sum > 21 and self.aces > 0):
            self.aces -= 1
            self.sum -= 10
        return 1 if self.sum <= 21 else 0
    
    def try_add_card(self, card):
        ## returns the result of adding the card to the sum without updating variables
        a = self.aces + 1 if card == 11 else 0
        s = self.sum + card
        while (s > 21 and a > 0):
            a -= 1
            s -= 10
        return s
    
    def has_bust(self):
        return 1 if self.sum > 21 else 0
    
    def print(self):
        print("Sum: %i; Aces: %i;" %(self.sum, self.aces))