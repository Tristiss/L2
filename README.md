# Python Programme für Paderborner Physik Praktikum (3P) Versuch L2 im GP1c

Dieses Repository beinhaltet alle nötigen Programme von der Messwertaufnahme bis zur Auswertung der Daten für den Praktikumsversuch L2 des Paderborner Physikpraktikums. Dabei ist der Aufbau speziell für die Untersuchung des Gütefaktors über die Bandbreite optimiert (vor allem die Auswertung). In den Programmen gibt es viele Variablen die jeweils für die verwendete Hardware bzw. Dateipfade angepasst werden müssen.

## Messwertaufnahme
### Requirements
Notwendigen Packages: 
* time
* pandas
* numpy
* threading
* scipy
* tinkerforge
* pynput

Notwendige Programme:
* L2.0 (Main Programm)
* Static_Methods_L2

Bei der Messwertaufnahme werden mehrere Wellenfunktionen für verschiedene Rotationsgeschwindigkeiten aufgenommen. Die Wellenfunktionen werden nach jedem Messpunkt in einer csv Datei gespeichert. Dabei sollte nur der path angepasst werden, da die andere Datenstruktur in der Datenaufbereitung wiederverwendet wird. Anzumerken ist, dass das Programm über die Esc Taste ordnungsgemäß abgeschaltet werden kann, sprich der Motor wird nicht (durch ein disablen während einer Bewegung) beschädigt. Bei jeglichen Software Bugs sollte sich das Programm ebenfalls aufgrund der try-except-Blöcke selber ausschalten. Falls der Versuch durch die Trennung des Netzteils erwirkt wird, sollte sichergestellt sein, dass auch das Programm ausgeschaltet wird (möglichst über die Esc-Taste).

## Datenaufbereitung
### Requirements
Notwendige Packages:
* pandas
* numpy
* scipy
* matplotlib

Notwendige Programme:
* L2.Intermediate

Bei der Datenaufbereitung wird die Amplitude aller aufgenommenen Wellenfunktionen über die Built-In max Funktion und die scipy curve_fit Funktion ermittelt und in eine weitere csv Datei geschrieben. Das Programm verwendet hier alle im vorgegebenen Ordner liegenden Dateien (die keine Ordner sind). Die Datenaufbereitung kann dabei direkt für mehrere Messreihen durchgeführt werden.
Es ist anzumerken, dass die Ergebnisse des Fits eigentlich besser sein müssten als die der max Funktion, da diese hier eigentlich fehleranfällig sein müsste. Jedoch hat der Fit aus unbekannten Gründen (vmtl. meine Inkompetenz) nicht funktioniert. Deswegen werden später die Daten der max Funktion verwendet, welche erstaunlich gut sind.

## Auswertung
Notwendige Packages:
* pandas
* numpy
* scipy
* matplotlib

Notwendige Programme:
* L2_Auswertung (Main Programm)
* functions_for_eval

Das Auswertungsprogramm fittet die Lorentzkurve (Resonanzkurve) über einen ODR Fit an die Messdaten, dabei werden die x und y Unsicherheiten beachtet (im Gegensatz zu den Fits aus Origin). Dieser Fit wird dann weiterverwendet, um die Grenzfrequenzen, Bandbreite, Gütefaktor und Dämpfung zu bestimmen. Da ich den analytischen Weg die Unsicherheit der Grenzfrequenz zu bestimmen nicht kannte, wurde dafür eine Monte Carlo Unsicherheitsfortpflanzung verwendet. Letztlich werden mehrere Plots erstellt (und falls test_phase deaktiviert gespeichert) und ein für die Bilanz des Berichts geschriebene Konsolenausgabe geprintet. Die Plots sind nicht optimal und nicht sehr übersichtlich, weshalb dieser Teil noch überarbeitet werden sollte.
