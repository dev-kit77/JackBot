from Blackjack import Card, Deck

def spacer():
	print("-------------------------------------------------------------")

print("Blackjack Object Tester!")

spacer()

print("Card Test: Ace of Clubs")

#init test card
testCard = Card(0,0)

print("Card: " + testCard.getCard())

print("Suit: " + str(testCard.getSuit()))

print("Ace Value < 21: " + str(testCard.getValue(6)))

print("Ace Value > 21: " + str(testCard.getValue(11)))

spacer()

print("Deck Tests")

spacer()

#init test deck
testDeck = Deck()

print("Deck Lenth before cards dealt: " + str(testDeck.cardsRemaining()))

spacer()

#init counter
i = 1

#loop for all cards in deck
while testDeck.cardsRemaining() > 0:
	print("Deal " + str(i) + ": " + testDeck.deal().getCard())

	#increment counter
	i += 1

spacer()

print("Deck Lenth after all cards dealt: " + str(testDeck.cardsRemaining()))