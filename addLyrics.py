

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, USLT, Encoding
from mutagen.mp3 import MP3

def add_lyrics_to_mp3(mp3_path, lyrics, lang="eng"):
    # Load the MP3 file
    audio = MP3(mp3_path, ID3=ID3)

    # Add ID3 tag if not present
    if not audio.tags:
        audio.add_tags()

    # Create a USLT (lyrics) frame
    uslt = USLT(encoding=Encoding.UTF8, lang=lang, desc='Lyrics', text=lyrics)

    # Add it to the tags
    audio.tags.add(uslt)

    # Save the updated file
    audio.save()

    print(f"Lyrics added successfully to {mp3_path}")



def main():
    mp3_file = "C:\\Users\\indka\\Music\\pons\\c05.mp3"
    lyrics_text = """
   Ein Fall für die Polizei? Franziska könnte heulen. Es ist bereits halb drei Uhr morgens und sie ist allein unterwegs. Um diese Zeit, ihr ist mulmig zumute. Sie studiert im zweiten Semester Germanistik und liebt das aufregende Studentenleben in Tübingen. Gestern war das erste Stocharkanrennen, bei dem sie dabei war. Für viele ist diese Veranstaltung sehr wichtig. Es findet jedes Jahr an einem Donnerstag Tag Mitte Juni statt. Verschiedene Fachschaften und Studentenverbindungen machen dabei mit. Der Gewinner muss abends eine Feier organisieren und frei bierspendieren. Der Verlierer muss eine Flasche leber dran trinken. Da aber vorher niemand weiß wer gewinnt wird überall gefeiert. Franziska war mit einigen Kommilitonen auf der Party im Haus der gehen. Aber der Weg hinunter in die Altstadt führt durch ein finsteres Stückchen Wald. Also hat sie gewartet, bis einer ihrer Komilitonen sie begleitet. Sie hat gedacht, Georg könnte sie bis vor ihre Haustür bringen. Doch sobald sie in der Altstadt angekommen waren, schwang er sich auf seinen Fahrrad und fuhr mit einem vorlichen, also bis dann, tschüss, davon. Auf dem Marktplatz sind noch einige ziemlich betrunkene Menschen leer. Nach 200 Meter, dann hat sie es geschafft. Franziska wohnt zusammen mit einer Komelitonin in einer kleinen Wohnung mitten in der wunderschönen Altstadt. Alles ist bequem zu Fuß zu erreichen, die Uni, alle wichtigen Geschäfte und ihre Lieblingskneiben. Und immer ist viel los, aber jetzt um diese Zeit ist die Stadt wie ausgestorben. Endlich da, sie biegt um die Ecke in die kleine Sackgasse in der sie wohnt. Dort ist alles dumm. Nur aus einem Fenster im ersten Stock fällt ein Licht in die Gasse. Und da sieht sie ihn, den Mann. Regungslos steht er da und schaut zum Fenster hinauf. Franziska stockt der Atem, denn in der Hand hält der Mann etwas. Eine Waffe, eine Pistole. Der Mann hat sie anscheinend nicht bemerkt, denn er blickt weiter zum Fenster hinauf. Er ist nicht mehr leicht benutzt, der Mann dann seine Pistole. "Gönns ruhig Franziska, nur keine Panik", sagt sie sich. Und tatsächlich wird sie ruhiger. Noch einmal, ganz vorsichtig, damit er sie nicht bemerkt, linst sie um die Ecke. Der Mann steht immer noch da und rührt sich nicht. Leise schleicht sie zu einem dunklen Haus-Eingang. Hier kramt sie in ihrem Rucksack nach ihrem Handy. Das sagt, wir kommen sofort. Franziska kommt es wie Stunden vor, doch tatsächlich sind es nur ein paar Minuten, bis ein Streifenwagen vorfährt und ca. 10 Meter von ihr entfernt anhält. Erleichtert geht Franziska aus ihrem Versteck auf den Streifenwagen zu. 3 Polizisten, eine Frau und zwei Männer, steigen aus und schließen Geräuschlos die Autotüren. Der Fahrer legt dabei seinen Zeigefinger an die Lippen. Am besten sind jetzt alle ganz still. den Arm um die aufgeregte Franziska und bleibt mit ihr am Wagen stehen. Die beiden Herren hingegen schleichen sich vorsichtig und mit gezüchter Waffe in die Gasse. Franziska ist richtig schwindelig. Was ist wenn jemand verletzt oder gar getötet wird? Nicht auszudenken. Doch dann hören die beiden Frauen, keinen Schuss, sondern nur lautes Gelächter. Eine der Polizisten erscheint und winkt die beiden Taschenlampe in die Hand und meint, keine Angst, der tut ihnen nichts, schauen sie selbst. Lensam und vorsichtig geht Franziska an der Seite des Polizisten in die Gasse und leuchtet den Mann mit der Waffe an. Wie er start bleibt sie stehen. Mein Gott ist das peinlich. Die Polizisten können sich vor Lachen kaum noch halten, denn in der Gasse steht kein Mann mit einer Pistole, sondern ein Mann mit einer Eiswaffel in der sondern aus dünnem Holz. Es ist eine Werbefigur. Wahrscheinlich gehört er zu dem kleinen Kaffee, das hier in der kleinen Gasse vor kurzem eröffnet wurde. Man hatte wohl vergessen, ihn bei Ladenschluss reinzustellen. Franziska möchte vor Scham am liebsten im Erdboden versinken. Wie konnte ihr so etwas nur passieren? Was sollten jetzt die Polizisten von ihr denken? Und ist es nicht so, dass man sehr viel hier kommen die Tränen. Die Polizisten legt wieder den Arm um sie und meint, es ist alles gut, es ist doch niemandem etwas passiert. Franziska erzählt ihr unter Tränen, dass sie kein Geld hat, den Einsatz zu zahlen. Doch die Polizisten meint nur, es ist gut, dass sie uns gerufen haben. Manchmal wird die Polizei auch bei einem richtigen Notfall nicht geholt. Da ist es uns doch lieber, wir kommen einmal umsonst.
    """
    add_lyrics_to_mp3(mp3_file, lyrics_text)

    # Example usage
if __name__ == "__main__":
    main()