from tkinter import *
from bostonuber import *

root = Tk()
root.title('Rideshare Price Calculator')
root.option_add('*Font', 'Trebuchet')
root.geometry("1000x800")

img = PhotoImage(file=r"PriceVSLocation.png")
img1 = img.subsample(2, 2)

# setting image with the help of label
Label(root, image=img1).grid(row=0, column=4,columnspan=3, rowspan=14, pady=5)

def calculate():
    # gathering data
    textvar = StringVar(root)
    answer = Label(root, text=textvar, pady=20)
    answer.pack_forget()
    source = option1drop.get()
    dest = option2drop.get()
    hour = int(entry_f.get())
    cab_type = companydrop.get()
    places = ['North End', 'West End', 'Downtown', 'Beacon Hill', 'Chinatown', 'Back Bay', 'South End', 'Fenway',
              'Roxbury']
    companies = ['Lyft', 'Uber']
    source = places.index(source)
    dest = places.index(dest)
    cab_type = companies.index(cab_type)

    # predicting distance from source and destination
    distance_model = pickle.load(open('distance_model.sav', 'rb'))
    data = [[source, dest]]
    dist = distance_model.predict(data)

    # opening the saved model
    main_model = pickle.load(open('finalized_model.sav', 'rb'))
    lyft_models = [0, 1, 2, 3, 4, 5]
    uber_models = [6, 7, 8, 9, 10, 11]
    all_models = ['Shared', 'Lyft', 'LyftXL', 'Lux', 'Lux Black', 'Lux Black XL', 'UberPool', 'UberX', 'UberXL',
                  'Black', 'Black SUV', 'WAV']
    cheapest_price = 100
    cheapest_model = ''
    expensive_price = 0
    expensive_model = ''
    # finding each price
    list = []
    if cab_type == 0:
        for model in lyft_models:
            data = [[hour, source, dest, cab_type, model, dist, 1.0]]
            result = main_model.predict(data)[0]
            set = False
            for i in range(len(list)-1):
                if result < list[i][1] and set == False:
                    list.insert(i, [all_models[model], result])
                    set = True
                    break
            if set == False:
                list.append([all_models[model], result])
    elif cab_type == 1:
        for model in uber_models:
            data = [[hour, source, dest, cab_type, model, dist, 1.0]]
            result = main_model.predict(data)[0]
            set = False
            for i in range(len(list)-1):
                if result < list[i][1] and set == False:
                    list.insert(i, [all_models[model], result])
                    set = True
                    break
            if set == False:
                list.append([all_models[model], result])
    # updating prices
    txt = ''
    for minilist in list:
        txt = txt + '$' + str(minilist[1]) + '0 for ' + str(minilist[0]) + '\n'
    ans.config(text=txt)


label_a = Label(master=root, text='\nPredicting Your Ride Price\n', font='Trebuchet 16 bold')
label_a.grid(row=0, column=2)
label_b = Label(master=root, text='Where are you departing from?')
label_b.grid(row=1, column=2)

places = ['North End', 'West End', 'Downtown', 'Beacon Hill', 'Chinatown', 'Back Bay', 'South End', 'Fenway', 'Roxbury']
option1drop = StringVar(root)
option1drop.set("Select Departure")
option2drop = StringVar(root)
option2drop.set("Select Destination")

drop1 = OptionMenu(root, option1drop, *places)
drop1.grid(row=2, column=2)

spacer2 = Label(master=root, text='\n')
spacer2.grid(row=3, column=2)

label_e = Label(master=root, text='What is your destination?')
label_e.grid(row=4, column=2)

drop2 = OptionMenu(root, option2drop, *places)
drop2.grid(row=5, column=2)

spacer3 = Label(master=root, text='\n')
spacer3.grid(row=6, column=2)

label_e = Label(master=root, text='Which rideshare company?')
label_e.grid(row=7, column=2)
options = ['Lyft', 'Uber']
companydrop = StringVar(root)
companydrop.set("Select Rideshare Company")
drop3 = OptionMenu(root, companydrop, *options)
drop3.grid(row=8, column=2)

label_f = Label(root, text='\n\nOther Info:')
label_f.grid(row=9, column=2)

label_g = Label(master=root, text='Temp (F):')
entry_d = Entry(master=root, width=7)
label_h = Label(master=root, text='# Passengers:')
entry_e = Entry(master=root, width=7)
label_i = Label(master=root, text='Hour (Military):')
entry_f = Entry(master=root, width=7)
label_g.grid(row=10, column=1, padx=30)
entry_d.grid(row=11, column=1, padx=30)
label_h.grid(row=10, column=2)
entry_e.grid(row=11, column=2)
label_i.grid(row=10, column=3)
entry_f.grid(row=11, column=3)

label_h = Label(master=root, text='\nWeather:')
label_h.grid(row=12, column=2)
weather = [' Overcast ', ' Foggy ', ' Clear ', ' Rain ', ' Snow ', 'Cloudy']
weatherdrop = StringVar(root)
weatherdrop.set("Select Weather")
drop4 = OptionMenu(root, weatherdrop, *weather)
drop4.grid(row=13, column=2)

spacer4 = Label(master=root, text='\n')
spacer4.grid(row=14, column=2)

Button(root, text="Calculate Price", padx=10, pady=5, command=calculate).grid(row=15, column=2)

ans = Label(master=root, text=' ', pady=30)
ans.grid(row=16, column=2)

root.mainloop()