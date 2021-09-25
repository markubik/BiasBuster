from model import HateSpeechModel
import time

def test_model():
    text = '''New York (CNN Business) President Joe Biden gave a gift to every major company in America by forcing them to mandate vaccines or stringently test their employees for Covid. Their reaction to the new rule: glee.
    Corporate America had been trying to navigate two competing pandemic realities: Companies are desperately trying to get back to business as usual, and mandating vaccines is among the best ways to accomplish that. But a labor shortage had tied their hands, as businesses have been worried that forcing people to get the shot would send some desperately needed employees and potential new hires packing.
    Some state and local governments had imposed various vaccine mandates, others had outright banned them — and all the while vaccines have also become politically charged.
    All that made universal corporate vaccine mandates difficult for employers. But Biden solved that problem for them this week.
    The Biden administration's new rule puts every American company with 100 or more employees in the same boat: All must get tough on their workers. If big companies want to do business in America, they have to help stop Covid from spreading.
    That gives the Walmarts of the world the arsenal they need to fix their biggest pandemic quandary. Employees have to get vaccinated or frequently tested, and companies don't have to worry about losing them to a rival.
    "While many companies will challenge the plan in courts, many others will likely let out a sigh of relief," said John Challenger, CEO of outplacement services company Challenger Gray & Christmas. "One of the most difficult problems for companies to tackle during this pandemic, among many, was the ever-changing and sometimes conflicting directives from local, state, and federal agencies."
    Praise from America's businesses
    Corporate America welcomed the news — most notably the Business Roundtable, an influential group of huge American companies led by Joshua Bolten, former chief of staff to President George W. Bush.
    "Business Roundtable welcomes the Biden Administration's continued vigilance in the fight against Covid," Bolten said in a statement. "America's business leaders know how critical vaccination and testing are in defeating the pandemic."
    Although the National Association of Manufacturers said it hoped the order would not disrupt their operations, the business group largely embraced Biden's new rule.
    "Getting all eligible Americans vaccinated will, first and foremost, reduce hospitalizations and save lives," said National Association of Manufacturers CEO Jay Timmons in a statement. "But it is also an economic imperative in that our recovery and quality of life depend on our ability to end_time this pandemic."
    Even the Chamber of Commerce, a frequent foe of Democrats like Biden, said it would work to encourage its members to get on board with the new rule "to ensure that employers have the resources, guidance, and flexibility necessary to ensure the safety of their employees and customers and comply with public health requirements."
    The Consumer Brands Association, a consortium of 2,000 packaged goods brands, also praised the effort to vaccinate as many Americans as possible: "We look forward to working with the administration to increase vaccination rates of essential workers throughout the country," said the trade group's CEO Geoff Freeman in a statement.
    However, the Consumer Brands Association said it lacked — and demanded — clarity about how its members are supposed to enforce the new rules.
    "The devil is in the details," Freeman said. "Without additional clarification for the business community, employee anxieties and questions will multiply."
    To date, a slow response from companies
    Bolten and Timmons both applauded the fact that many companies implemented vaccine mandates on their own. But the response hasn't been enough, Biden warned. He said the pandemic is in danger of spiraling out of control and it's long past time for all major companies to take action to ensure their employees are safe.
    "We've been patient, but our patience is wearing thin," Biden said in a speech announcing the new rules Thursday. "The time for waiting is over."
    Although many companies have required some of their employees to be vaccinated, few ensure their entire staff is vaccinated.A Willis Towers Watson survey of 1,000 American companies released last week found that just more than half (52%) of businesses considered implementing vaccine mandates by the end_time of 2021, but only 21%% had them in place already.
    "The majority of employers have simply been encouraging their teams to get vaccinated, knowing a vaccinated workforce would significantly reduce the chances of outbreaks and, therefore, work stoppages," said Challenger. "The political climate surrounding vaccines made it difficult for employers to mandate them, as no company wants to get into the crosshairs of political activist groups or alienate their team members."
    What it means for small businesses
    Although the Biden administration's vaccine rule doesn't apply to small businesses, it could give some of them cover to mandate vaccines as well. Some smaller companies may have worried about losing vaccine-hesitant workers to a Costco (COST) or Walmart if they mandated the vaccine. Now that may no longer be a risk.
    But the rule could also exacerbate America's labor shortage, Challenger argues. Some vehement anti-vaccine workers may move to smaller companies that look to take advantage of their exclusion from Biden's new order. Other mid-sized companies may lay off some workers to get under the 100-employee cap. That could limit some job-seekers' prospects.
    Still, the bottom line is clear: The majority of American workers — two-thirds of them — will have to either receive the vaccine, get frequently tested or find work someplace else. That's a blessing to the companies that employ them, which have been long searching for the cover the Biden administration just provided them.'''

    model = HateSpeechModel()
    print('Loading model...')
    start_time = time.time()
    model.load()
    end_time = time.time()
    print('Model loaded in', end_time-start_time, 'seconds')
    start_time = time.time()
    prediction = model.predict(text)
    end_time = time.time()
    print('Model predicted:', prediction, 'in', end_time-start_time, 'seconds')

    assert(prediction == 'OFFENSIVE')


if __name__ == '__main__':
    test_model()