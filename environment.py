import random
from actor import Actor
from copy import copy

class Environment:
    def __init__(self, min, max, number_of_decks, 
                 living_cost=-0.01,
                 progress_reward=0.02,
                 natural_blackjack_bonus=0.5):
        self.living_cost = living_cost
        self.progress_reward = progress_reward
        self.natural_blackjack_bonus = natural_blackjack_bonus
        self.deck = [x * number_of_decks for x in (4, 4, 4, 4, 4, 4, 4, 4, 16, 4)]
        self.score = 0 # the card counting "score" of the deck
        # removes a random number of cards from min to max
        for i in range(random.randint(min, max)):
            x = self.draw()
        self.dealer = Actor(self.draw())
        self.player = Actor(self.draw(), self.draw())

        # track if episode already ended
        self.done = False

        # Reward immediate natural 21
        self._apply_natural_blackjack_if_any()

    def draw(self):
        ## gets a card from the deck and updates the score and deck
        x = random.sample(range(10), counts=self.deck, k=1)[0]
        self.deck[x] -= 1
        if x > 7:
            self.score -= 1
        elif x < 5:
            self.score += 1
        return x + 2
    
    def peek(self):
        ## gets a card from the deck without affecting the score or deck
        return random.sample(range(10), counts=self.deck, k=1)[0] + 2
    
    def undraw(self, card):
        ## returns a given card to the deck and updates the score
        if card > 9:
            self.score += 1
        elif card < 7:
            self.score -= 1
        self.deck[card - 2] += 1
    
    def getscore(self):
        return self.score
    
    def print(self):
        print("Cards remaining: %s" %" ".join([str(x) for x in self.deck]))
        print("Dealer: %i" %self.dealer.sum)
        print("Player: %i" %self.player.sum)

    def probability_of_busting(self):
        ## probability of drawing a card that would give a score higher than 21
        ## returns numerator, denominator, quotient
        events = sum(self.deck[max(-1, 22 - self.player.sum - 2):-1])
        total = sum(self.deck)
        return events, total, events / total

    def probability_of_not_busting(self):
        ## probability of drawing a card that would give a score lower than 21
        ## returns numerator, denominator, quotient
        result = self.probability_of_busting()
        return result[1] - result[0], result[1], 1 - result[2]
    
    def remaining_cards(self):
        return sum(self.deck)
        
    def _apply_natural_blackjack_if_any(self):
        # If player starts at 21, treat as terminal.
        # (Dealer behavior is simplified since you currently only deal one dealer card initially.)
        if self.player.sum == 21:
            # If dealer already has 21 with one card (only possible if 11)
            # treat as draw; otherwise win + bonus.
            self.done = True
            
    def _apply_natural_blackjack_if_any(self):
        # If player starts at 21, treat as terminal.
        # (Dealer behavior is simplified since you currently only deal one dealer card initially.)
        if self.player.sum == 21:
            # If dealer already has 21 with one card (only possible if 11)
            # treat as draw; otherwise win + bonus.
            self.done = True
    
    def hit(self):
        return self.player.add_card(self.draw)

    def hit_verbose(self):
        print("Player hits: %i -> " %self.player.sum, end="")
        self.player.add_card(self.draw())
        print(self.player.sum)
        return self.player.has_bust()
    
    def try_hit(self):
        # hits without updates variables
        x = self.peek()
        return 1 if self.player.try_add_card(x) > 21 else 0
    
    def try_hit_count(self, count):
        # try_hit's multiple times and prints success rate
        busts = 0
        for i in range(count):
            busts += self.try_hit()
        return (busts, count, busts / count)

    def stand(self):
        ## returns -1 on loss, 0 on draw, 1 on win
        self.dealer.add_card(self.draw())
        while self.dealer.sum < 17:
            self.dealer.add_card(self.draw())
        if (not self.player.has_bust() and self.player.sum == self.dealer.sum): return 0 # draw
        elif (not self.player.has_bust() and (self.player.sum > self.dealer.sum or self.dealer.has_bust())): return 1 # win
        else: return -1 # loss
        
    def stand_verbose(self):
        print("Dealer draws: %i" %self.dealer.sum, end="")
        while self.dealer.sum < 17:
            self.dealer.add_card(self.draw())
            print(" -> %i" %self.dealer.sum, end="")
        print("")
        if (not self.player.has_bust() and self.player.sum == self.dealer.sum):
            print("%i vs. %i, draw!" %(self.player.sum, self.dealer.sum))
            return 0
        if (not self.player.has_bust() and (self.player.sum > self.dealer.sum or self.dealer.has_bust())):
            print("%i vs. %i, player wins!" %(self.player.sum, self.dealer.sum))
            return 1
        else:
            print("%i vs. %i, dealer wins!" %(self.player.sum, self.dealer.sum))
            return -1
        
    def try_stand(self):
        # tries to stand without affecting any fields
        # awful lot slower and more memory intensive than try_hit
        # since we need to store the cards drawn and a copy of the dealer
        temp_dealer = copy(self.dealer)
        drawn = []
        while (temp_dealer.sum < 17):
            x = self.draw()
            temp_dealer.add_card(x)
            drawn.append(x)
        [self.undraw(x) for x in drawn]
        if (not self.player.has_bust() and self.player.sum == self.dealer.sum): return 0 # draw
        elif (not self.player.has_bust() and (self.player.sum > self.dealer.sum or self.dealer.has_bust())): return 1 # win
        else: return -1 # loss

    def try_stand_count(self, count):
        successes = 0
        for i in range(count):
            successes += 1 if self.try_stand() == 1 else 0
        return (successes, count, successes / count)

    def observe(self):
        # returns the state to be given to an agent
        return (self.player.sum, self.player.aces, self.dealer.sum, self.dealer.aces, self.score)

    def step(self, action: int):
            """
            action:
              0 = stand
              1 = hit
    
            returns: (obs, reward, done, info)
            """
            if self.done:
                return self.observe(), 0.0, True, {"msg": "episode already done"}
    
            reward = float(self.living_cost)  # small step penalty to discourage endless play
            info = {}
    
            if action == 1:  # hit
                prev_sum = self.player.sum
                self.player.add_card(self.draw())
    
                if self.player.has_bust():
                    # bust is terminal loss
                    self.done = True
                    reward += -1.0
                    info["terminal"] = "bust"
                else:
                    # small positive shaping for improving hand total (but not too large)
                    if self.player.sum > prev_sum:
                        reward += self.progress_reward
    
                    # optional: if you hit to 21, end immediately as a "stand-equivalent"
                    # (this usually helps learning)
                    if self.player.sum == 21:
                        # resolve dealer and outcome
                        outcome = self.stand()
                        if outcome == 1:
                            reward += 1.0
                        elif outcome == 0:
                            reward += 0.0
                        else:
                            reward += -1.0
                        info["terminal"] = "reached_21"
    
            elif action == 0:  # stand
                outcome = self.stand()  # -1/0/1 terminal
                if outcome == 1:
                    reward += 1.0
                    # bonus if player had a natural 21 at start (optional)
                    if self.player.sum == 21:
                        reward += self.natural_blackjack_bonus
                elif outcome == 0:
                    reward += 0.0
                else:
                    reward += -1.0
                info["terminal"] = "stand"
    
            else:
                # invalid action: punish and end (or you can keep running; ending simpler for training stability)
                self.done = True
                reward += -1.0
                info["terminal"] = "invalid_action"
    
            return self.observe(), float(reward), bool(self.done), info
        def to_dict(self):
            return {
                "deck": list(self.deck),
                "score": int(self.score),
                "dealer": self.dealer.to_dict(),
                "player": self.player.to_dict(),
                "done": bool(self.done),
                "living_cost": float(self.living_cost),
                "progress_reward": float(self.progress_reward),
                "natural_blackjack_bonus": float(self.natural_blackjack_bonus),
                "random_state": random.getstate(),
            }
        @classmethod
        def from_dict(cls, data: dict):
            # construct without burning cards: use dummy init then overwrite state
            obj = cls(min=0, max=0, number_of_decks=1,
                      living_cost=data.get("living_cost", -0.01),
                      progress_reward=data.get("progress_reward", 0.02),
                      natural_blackjack_bonus=data.get("natural_blackjack_bonus", 0.5))
    
            obj.deck = list(data["deck"])
            obj.score = int(data["score"])
            obj.dealer = Actor.from_dict(data["dealer"])
            obj.player = Actor.from_dict(data["player"])
            obj.done = bool(data.get("done", False))
    
            # restore RNG last so subsequent draws match
            rs = data.get("random_state", None)
            if rs is not None:
                random.setstate(rs)
    
            return obj
    
        def save(self, path: str):
            with open(path, "wb") as f:
                pickle.dump(self.to_dict(), f)
    
        @classmethod
        def load(cls, path: str):
            with open(path, "rb") as f:
                data = pickle.load(f)
            return cls.from_dict(data)
