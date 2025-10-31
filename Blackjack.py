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

	def clear(self):
		#reset score
		self.__score = 0

		#return card array for discard
		return self.__cards

class Player(Agent):
	def __init__(self):
		super().__init__()

class Dealer(Agent):
	def __init__(self):
		super().__init__()

class Table:
	def __init__(self):
		self.__cards = Deck()
		self.__discard = list()
		self.__state = 0

	def bet(self, player):
		pass

	def hit(self, agent):
		pass

	def stand(self, agent):
		pass

	def doubleDown(self, player):
		pass

	def split(self, player):
		pass

	def surrender(self, player):
		pass
		