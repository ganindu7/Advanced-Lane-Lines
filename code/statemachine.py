

class StateMachine():
    def __init__(self):
        self.handlers = {}
        self.startingState = None
        self.exitStates = []

    def addNewState(self, name, handler, exit_state=0):
        name = str.upper(name)
        self.handlers[name] = handler # passing function object to handler 
        if exit_state:
            self.exitStates.append(name)

    def setStartingState(self, name):
        self.startingState = str.upper(name)

    def run(self, image):
        try:
            handler = self.handlers[self.startingState]
        except:
            raise AttributeError('must call method: setStartingState() before run()')

        if not self.exitStates:
            raise AttributeError('There must be at least one exit state')

        while 1:

            '''
            In this case a whole run spits out a result image
            Configs settings for the next image based on the output
            '''
            
            (newState, payload) = handler(self, image)

            if str.upper(newState) in self.exitStates:
                break
                
            else:
                handler = self.handlers[str.upper(newState)]











