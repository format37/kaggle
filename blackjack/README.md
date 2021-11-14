# blackjack
I don't know why blackjack.simulate() and blackjack.simulate_one_game() does not return any value. All that we lack for training is reward point of game.
So. My decision is a little modify learntools to return reward point. And then train the model, print the list and copy&paste to kaggle notebook.
1. clone https://github.com/Kaggle/learntools/
2. make a new project in /learntools/
Now we need to find simulate_one_game function and then modify them. To search is conveniently to use github search by project.
3. add return before game.play() of simulate_one_game(self, phit): in learntools/learntools/python/ex3.py
4. also be useful to disable log messages while training. Go to learntools/learntools/python/blackjack.py find def log(self, msg): and change print(msg) to pass#print(msg)
Unfortunately i don't see the native way to solve this competition. Anyway this works B)
