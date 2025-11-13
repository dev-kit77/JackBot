import random

class Card:
	def __init__(self, suit, card):
		self.__suit = suit
		self.__card = card

	def getValue(self, currScore):
		match self.__card:
			case 0:
				if (currScore + 11) > 21:
					return 1
				else:
					return 11
			case 10 | 11 | 12:
				return 10
			case _:
				return self.__card + 1
	
	def getSuit(self):
		return self.__suit

	def getCard(self):
		#init string for card descriptor
		cardString = ""

		#assign suit descriptor
		match self.__suit:
			case 0:
				cardString += "♣"
			case 1:
				cardString += "♦"
			case 2:
				cardString += "♥"
			case 3:
				cardString += "♠"

		#assign card type
		match self.__card:
			case 0:
				cardString += "A"
			case 10:
				cardString += "J"
			case 11: 
				cardString += "Q"
			case 12:
				cardString += "K"
			case _:
				cardString += str(self.__card + 1)

		#return card string representation
		return cardString

class Deck:
	def __init__(self):
		#init deckArray 
		self.__deckList = list()

		#init random no gen 
		random.seed()

		#loop for all suits
		for suit in range(4):
			#loop for all cards in suit
			for card in range(13):
				#add card to deck
				self.__deckList.append(Card(suit, card))

		#shuffle deck
		random.shuffle(self.__deckList)

	#remove card from top of deck
	def deal(self):
		return self.__deckList.pop(0)
	
	#check how many cards are remaining in deck
	def cardsRemaining(self):
		return len(self.__deckList)

class Agent:
	def __init__(self):
		#init defaut values
		self.__cards = list()
		self.__score = 0

	def addCard(self, draw):
		#add card to array
		self.__cards.append(draw)

		#reset score
		self.__score = 0

		#calc score
		for card in self.__cards:
			self.__score += card.getValue()
		
		#check for bust
		if self.__score > 21:
			return False
		else:
			return True
		
	def getScore(self):
		return self.__score

	def clear(self):
		#reset score
		self.__score = 0

		#return card array for discard
		return self.__cards

class Player(Agent):
	def __init__(self):
		super().__init__()
		#init player specific attributes
		self.__cash = 500
		self.__bet = 0

	def getCash(self):
		#return the current player cash pot
		return self.__cash
	
	def getBet(self):
		#return the current player bet
		return self.__bet
	
	def setBet(self, bet):
		#remove bet from cash pot
		self.__cash -= bet
		#set bet value
		self.__bet = bet

	def addCash(self, winnings):
		#add winnings to cash pot
		self.__cash += winnings

	def clear(self):
		self.__bet = 0
		return super().clear()

class Dealer(Agent):
	def __init__(self):
		super().__init__()

	def peekCard(self):
		return self.__cards.index(0)

class Table:
	def __init__(self):
		#init table fields
		self.__cards = Deck()
		self.__discard = list()
		self.__dealer = Dealer()
		self.__state = 0

	def bet(self, player, bet):
		#check if bet state
		if (self.__state == 0):
			#set player bet
			player.setBet(bet)
			
			#set table into game state
			self.__state = 1

			#return true for bet placed
			return True
		else:
			#return false for state error
			return False

	def hit(self, agent):
		#check result of dealt card
		if (agent.addCard(self.__cards.deal())):
			#card was dealt with no bust
			return True
		else:
			#return failure
			return False

	def stand(self, agent):
		#check if in play state
		if (self.__state == 1):
			#set state to dealer play
			self.__state = 2
			#return success
			return True
		elif (self.__state == 2):
			#set state to comparison state
			self.__state = 3
			#return success
			return True
		else:
			#failure as game is not in play
			return False

	def doubleDown(self, player):
		#check result of dealt card
		if (player.addCard(self.__cards.deal())):
			if (self.stand(player)):
				#card was dealt with no bust and player is now standing
				return True
			else:
				#return failure due to state error
				return False
		else:
			#return failure
			return False

	def surrender(self, player):
		#check if in play state
		if (self.__state == 1):
			#set state to betting state
			self.__state = 0

			#clear current player
			player.clear()

			#return success
			return True
		else:
			#failure as game is not in play
			return False
		