import joblib
import pandas as pd

class Game:
    """
    Game Class
    """
    def __init__(self, name, cpu, ram, battery, latency):
        self.name = name
        self.cpu = cpu
        self.ram = ram
        self.battery = battery
        self.latency = latency

    def handler(self):
        """
        Return the all the values of a game object
        """
        input_data = pd.DataFrame({
            'CPU_Usage' : [self.cpu],
            'RAM_Usage' : [self.ram],
            'Battery_Impact' : [self.battery],
            'Latency' : [self.latency]
            })
        return input_data
    
class Application:
    """
    Main Application
    """
    def __init__(self):
        self.games_running = {}
        self.model = joblib.load('model10.pkl')

    def create_games_object(self, name, cpu, ram , battery, latency):
        """
        Create a new game object and predict the usage of the game
        """
        new_game = Game(name=name, cpu=cpu, ram=ram, battery=battery, latency=latency)
        new_game_data = new_game.handler()
        pred_data = self.model.predict(new_game_data)
        self.games_running[new_game.name] = (new_game, pred_data)

    def display(self):
        """
        Display on the GPU usage of every game running in the system
        When there a game requiring 30% GPU usage is running, the model will calculate the appropriate usage amount and allocate to it.
        When there is more than one game running, it will balance the usage among those games to optimize the resource allocation.
        """
        tab = "  "
        ret_val = "  Name  |  CPU_USAGE  | RAM_USAGE  |  BATTERY  |  LATENCY  |  GPU_USAGE  \n"
        games_running = self.games_running.values()
        for game, gpu_usage in games_running:
            ret_val += tab + game.name + tab + "|" +\
                       tab + str(game.cpu) + tab + "|" +\
                       tab + str(game.ram) + tab + "|" +\
                       tab + str(game.battery) + tab + "|" +\
                       tab + str(game.latency) + tab + "|" +\
                       tab + str(round(gpu_usage[0] * 100)) + tab + "\n"
            
        print(ret_val)

if __name__ == '__main__':
    app = Application()
    app.create_games_object("A", 5.121085, 99.646115, 0.937204, 151.423450)
    app.display()
