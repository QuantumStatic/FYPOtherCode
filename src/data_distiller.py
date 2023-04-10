import csv
import math
from datasets import load_dataset, dataset_dict
from myfunctions import execute_this, clear_output_screen
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

DATA_FOLDER_PATH = "C:\\Users\\uujain2\\Desktop\\Utkarsh\\FYP\\Dataset\\data"
BIG_DS_PATH = f"{DATA_FOLDER_PATH}\\BIG_DS"

def creae_data_dict() -> dict[str, list]:
    """
    Helper function to convert the datasets to Huggingface supported format
    """
    data_dict = {'texts': [], 'labels': []}
    with open(F"{DATA_FOLDER_PATH}\\final_labels_MBIC_new.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        file_iter = iter(csv_reader)
        next(file_iter)
        for row in file_iter:
            data_dict['texts'].append(row[0])
            label_val = -1
            match row[7]:
                case "Biased":
                    label_val = 1
                case "Non-biased":
                    label_val = 0
                case "No agreement":
                    label_val = 2
            data_dict['labels'].append(label_val)

    return data_dict

def create_and_save_distilled_dataset(train_texts:list[str], test_texts:list[str], validation_texts:list[str], train_labels:list[int], test_labels:list[int], validation_labels:list[int]):
    """
    Helper function to save the distilled dataset in csv format. All 3 datasets are saved in their corresponding files.
    """
    with open(f"{DATA_FOLDER_PATH}\\train_t5.csv", "w", newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
        # csv_writer.writerow(['texts', 'labels'])
        # csv_writer.writerows(iter(train_texts))
        csv_writer.writerows(zip(train_texts, train_labels))

    with open(f"{DATA_FOLDER_PATH}\\test_t5.csv", "w", newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
        # csv_writer.writerow(['texts', 'labels'])
        # csv_writer.writerows(iter(test_texts))
        csv_writer.writerows(zip(test_texts, test_labels))

    with open(f"{DATA_FOLDER_PATH}\\validation_t5.csv", "w", newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',')
        # csv_writer.writerow(['texts', 'labels'])
        # csv_writer.writerows(iter(validation_texts))
        csv_writer.writerows(zip(validation_texts, validation_labels))

def experiment_huggingface():
    """
    Experiment to see how the huggingface datasets library works and a way to tokenize it
    """
    dataset = load_dataset('csv', data_files={'train': f"{DATA_FOLDER_PATH}\\train.csv", 'test': f"{DATA_FOLDER_PATH}\\test.csv", 'validation': f"{DATA_FOLDER_PATH}\\validation.csv"}, column_names=['text', 'label'], split=['train', 'test', 'validation'])
    clear_output_screen()
    # print(dataset[0][200])
    dataset = dataset_dict.DatasetDict({'train':dataset[0], 'test':dataset[1], 'validation':dataset[2]})
    # print(dataset['train'][200])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True), batched=True)
    print(dataset)


def testing():
    string= """texts,labels
"Bernie Sanders Is Crazy, But The Only Democrat Candidate Voters Actually Like",1
"In July 2019, Rep. Alexandria Ocasio-Cortez, D-N.Y., and other Democrats visited a similar processing center in Clint, Texas, and made explosive allegations about the conditions there.",1
U.S. President Donald Trump said on Wednesday a tax cut for the middle class would be announced over the next 90 days.,0
A leader of a movement that ur444444444444444444444444444444444444444out having an abortion promoted the procedure to children in a video from Cut’s HiHo on the YouTube channel.,1
The ballooning cost of college and massive student loan debt is recognized as reaching crisis levels by politicians on both sides of the aisle.,0
Hundreds of cops packed a California church Saturday in final salute to police officer Ronil Singh who was killed the day after Christmas when he stopped an illegal immigrant suspected of driving drunk.,0
"Today’s mainstream media doesn’t want you to remember, but the early days of feminism were rife with ideological battles and  backstabbing that would put a medieval court to shame.",1
"Trump's latest thinking on the task force, created in January to coordinate the response to the virus, came a day after Vice President Mike Pence and other officials said the group's work would be transferred to federal agencies as soon as this month. The idea drew fire from critics who said it was too soon to disband the group.",0
"Chinese state media outlets have been bursting with criticism for President Donald Trump during the current round of riots, but the Chinese Communist Party (CCP) also tossed a few bricks at former President Barack Obama and his Democrat Party, needling them for doing little about racial issues and touting China’s efforts to care for its poor and disenfranchised as vastly superior to the Democrats’ half-hearted socialism.",1
"This idea, gaining traction under the hashtag #CancelStudentDebt following the release of Sanders’ college debt forgiveness plan, is bad for several reasons, but naturally appealing in a sort of populist way, since plenty of people choose not to think about second-order impacts and potential consequences of driving our country into even more ruinous national debt than before.",1
Democrats have seethed as they helplessly watched the Trump administration undo or block a slew of Obama-era environmental proposals during the past two years.,1
Her call for a Green New Deal -- which would seek to focus on income inequality and climate change by funding a host of radical ideas on liberal wish-lists and overhaul the nation's energy sector -- has been embraced by a number of 2020 hopefuls already.,1
Since the Democrat Congress and the Obama administration orchestrated the government takeover of student loans in 2010 the total amount of student loan debt has exploded.,1
The Trump administration is working on a plan to let the Social Security Administration (SSA) check up on claimants on Facebook and Twitter in order to root out fraud and abuse in the disability program.,0
"Climate change, which is the subject of fierce debate, may lead to a ""substantial increase"" in the number of extreme rainstorms, according to a startling new study by NASA.",1
Oprah Schools Instagram Hater With Receipts After Student Loan Diss,1
"An illegal alien has been charged with enticing a 13-year-old girl through social media and raping her in Madison County, Alabama.",1
Democrats also said some girls might risk being thrown out of their homes or beaten if they tell their parents they're pregnant.,0
"Mafia exploitation of immigrants on farms in Italy is likely to rise as the coronavirus lockdown brings chronic labor shortages, the government and labor rights groups have warned.",1
"Sen. Cory Booker, D-N.J., defended his support of the controversial “Green New Deal” on Friday, by comparing the government-led push to overhaul the nation’s economy and energy sector to landing on the moon and defeating the Nazis in World War II.",2
"Before the shooting death of women's rights activist and artist Isabel Cabanillas de la Torre in Ciudad Juarez, Mexico, human rights attorneys and others already had noted several years of rising femicides in the border city.",0
"China – the country that throws its minorities into concentration camps and uses them for slave labor, the brutal authoritarian regime driven by Han Chinese racial supremacist ideology, the government that thinks nothing of wiping out entire villages if it needs to build a few dams quickly to put on a good show for the Olympics – boasted of its superior compassion and tolerance in the pages of the CCP’s Global Times on Monday",1
Officials at Dartmouth College looked the other way this week when a group of leftist student activists disrupted classes to bring attention to their support for the Green New Deal.,1
"Proponents of better compensation and labor rights for college athletes have hammered the NCAA on numerous fronts over the last decade, with federal lawsuits, attempts to unionize, legislation in statehouses and Congress, and new leagues that aim to compete for top-tier athletes by paying them more than the NCAA allows.",2
"The government wants to usurp parents, any chance they get. What consequences does this have in the lives of American families?",1
"Google declined to offer details beyond Huntley’s tweets, but the unusually public attribution is a sign of how sensitive Americans have become to digital espionage efforts aimed at political campaigns.",0
Assad has been at war with the rebels since 2011 and has been repeatedly accused of using chemical weapons against civilians and other human rights atrocities to cement his control over regions that oppose his rule.,1
Bouchard commented on his efforts to secure state preemption of the myriad gun-free zones in the state.,0
"Of course, billionaires and corporations that reaped millions of dollars from the GOP tax cuts pumped some of that windfall into Republican campaigns, allowing Sen. Mitch McConnell, R-Ky., to retain control of the Senate.",1
"Bryce Harper, who signed the richest free-agent contract in baseball history, a cool $330 million, was being unveiled for the first time in a Phillies uniform.",0
The GOP controls both the legislature and the Michigan State Capitol Commission — the two entities that could enact a ban — and neither appears willing to act.,2
Strategic purchases of U.S. businesses and the placement of Chinese companies on American stock exchanges and indexes have also given the PRC enormous suasion over the avenues of American soft power.,1
"Of course, the demands of tolerance are not universal. The strong gods are still suppressed and exiled, and the heretic-hunting inquisitors of diversity have become more paranoid and vindictive in triumph.",1
"Of course, social distancing will be inconceivable for football players and the results all too predictable in a world in which Covid-19 is likely to hang around for a long, long time. ",1
"Trans sportswomen hit back at Martina Navratilova, after the tennis champion said “its insane and it’s cheating” for transgender women to be allowed to compete in women’s sport.",0
"Today, the cumulative stress of Donald Trump’s escalating trade war is leading to if not an irreparable breach between America and China, then certainly a significant fraying.",1
"Transgender controversy also erupted ahead of International Women’s Day in Leicester, after a students’ union at Leicester University voted to rebrand the day as International Womxn’s Day, in order to be more inclusive of transgender women.",0
A Texas law requires contractors who work for or do business with the state to certify that they do not boycott Israel or Israeli-occupied territories.,0
"Some of Europe's biggest clubs, such as FC Barcelona in Spain and Juventus in Italy — two of the hardest-hit countries — have already agreed to hefty wage cuts with their stars.",2
"The anti-vaccine movement made headlines last spring, when widespread distrust of the pharmaceutical industry and the belief that vaccines can lead to autism in children led to a measles outbreak. ",1
"Andres Iniesta, the former Barcelona captain and World Cup-winning Spanish soccer star, apologized Tuesday after being criticized for posting a picture of him posing with two people clad in blackface during a popular Christian celebration.",0
"Hungarian Prime Minister Viktor Orbán also denounced the policy of Chancellor Angela Merkel and the attitude of the German media, which “are constantly pushing” for Europe to welcome more and more migrants, in opposition to the will of the Hungarian people.",0
"In Philadelphia, there were also reports that the city was ignoring social distancing standards by cramming their new voting machines (one per jurisdiction) into multi-precinct voting locations, which forced voters and poll workers to stand on top of each other.",1
"The SPLC last year released hundreds of emails of Stephen Miller,the architect of Trump’s immigration policies such as the child-parent separations at the border and the Muslim ban, in which he cited and promoted ""explicitly white nationalist websites."",1
"Trump was a major factor in the early rumblings of white nationalism on the network, which gave him considerable airtime during his reality TV days to air conspiracy theories about Barack Obama’s birth certificate.",1
"President Donald Trump asserted without evidence Friday that the COVID-19 pandemic will ""go away without a vaccine."" The comments contradict the belief of some of the president's task force members.",1
Democratic lawmakers and environmental groups say the changes will exempt polluters from public scrutiny of their projects.,0
"That the Secret Service takes precautions to protect the president in a time of chaos is, in itself, to be expected. But the contrast between Trump’s chest-thumping and the fact that he was cowering in the basement was objectively funny.  Unsurprisingly, a lot of people made fun of him, and he didn’t like it.",1
"Gov. Tate Reeves, who has worked for years to limit and end abortion in Mississippi, promised to take action against the state's lone abortion clinic if it continues to provide abortions during the coronavirus pandemic.",0
"The U.S. states of Texas and Ohio have ordered abortions be postponed as non-essential procedures to free up resources to fight coronavirus, a move critics said on Tuesday was political.",0
"As a political calculation, it shows Biden’s campaign is confident about their standing with many suburban moderates who were swing voters in the past but have moved solidly into the Democratic Party during Trump’s presidency, offended by his bullying demeanor and chaotic style.",1
"Even if you don’t speak white nationalist dog whistle, this speech is pretty shocking. Hawley is claiming that the United States is run, and has been run, by a secret group of international “elites” who value terrible things—such as education, achievement, and progress.",1
"Trump appeared to be referring to Northam's signing of gun control measures at the beginning of April, moves which drew condemnation from Republicans and criticism from Trump. ",0
"House Speaker Nancy Pelosi (D-CA) on Monday waxed hyperbolic about the 2020 presidential election, telling an audience that “civilization as we know it is at stake.”",1
"One of the things that makes Alabama’s measure extraordinary is its radical simplicity: it simply bans all abortions, at every stage of pregnancy.",1
Germany tries to stop Trump from luring away firm working on coronavirus vaccine.,2
"Athletes and fans are excited for Sowers' historic role at Super Bowl LIV and have remarked that her position is especially poignant given there have only been 13 out gay and bi players in the NFL's 100-year history, according to OutSports.",1
Court filings show the NRA is in shambles — and Wayne LaPierre hopes his lawyer can ‘keep him out of jail’,2
President Donald Trump has picked his favorite weapon to fight the economic fallout from the coronavirus: an elimination of the “payroll tax” on workers’ gross earnings that is used to fund national retirement programs.,1
"Numerous posts claim the virus has been patented — and some even suggest, incorrectly, that the virus was made in a lab and a vaccine already exists.",2
He then criticized the attempt to lump Republicans and white nationalists into the same group.,1
"Not only do most U.S. workers not have that luxury – 29% have the option to work remotely, according to Labor Department data – many have frontline exposure to the virus because their jobs put them in front of people.",0
Power is pretty much all the racist right wants.,1
White people are being asked—or pushed—to take stock of their whiteness and identify with it more. This is a remarkably bad idea.,1
"Many conservatives came to Hawley’s defense, slamming Tapper’s attack on the Missouri senator.",0
The harshest attack on Trump and military action against civilians was issued in The Atlantic on Wednesday by former Defense Secretary and retired Marine Corps general Jim Mattis. He hailed peaceful protesters for working for a better future.,1
"However, Democrats, including some of the 2020 candidates, have continued to push a narrative that Trump called white supremacists “very fine people,” when he clearly did not.",1
"Justice Sonia Sotomayor, in her dissent, accused the majority of weaponizing the First Amendment — an unconscionable position for a person tasked with “faithfully and impartially” discharging the duty to protect the inherent rights of all Americans. The dissenters have perverted the intent of the Founders. It’s none of Sotomayor’s business how we use free expression. It’s her business to protect it.",1
"If sports don’t trump religion as the opiate of the masses, they have, until recently, been at least the background music of most of our lives.",1
The number of April background checks to purchase a firearm showed Americans flocking to gun stores in high numbers for the second month in a row amid worries about the coronavirus pandemic.,1
"Omar believes USA Powerlifting is discriminating against males who identify as transgender women, based on the “myth” that they have a “direct competitive advantage” over biological females.",1
"After the airstrike that killed Iranian Gen. Qasem Soleimani, there were reports that U.S. Customs and Border Patrol detained dozens of people at the Blaine, Washington, port of entry over the weekend.",0
"On top of the enormous financial impact, these beauty regimens require these women to take the time out of their busy schedules to attend to their appearances in a way that men in power are spared. ",0
"But there’s also an intensifying global climate crisis, which carries life-changing risks for billions of people.",1
"In June, an explosive early morning fire rocked the Philadelphia Energy Solutions refinery, terrifying nearby residents.",0
"Friday, a legal services organization at Yale Law School sent a letter to the high court urging that the administration's decision to terminate DACA should be blocked in light of the pandemic.",0
"Anti-vaxxers represent only about 2 percent of American families, although you’d be forgiven for thinking they’re more numerous. They certainly make a lot of noise and have some world-famous adherents. However, like malaria-carrying mosquitoes, this tiny group that punches above its weight may be on the verge of creating a public health problem for us all.",1
"The Trump administration and Republicans in Congress have stepped up their efforts to hold China accountable for intellectual property theft, unfair trade practices, and aggression towards U.S. allies in East Asia, as it seeks to replace the U.S. as the world’s superpower by 2049.",0
"Now, with Democrats in control of the House of Representatives after the midterm elections, and with states like New York and Virginia among others pushing boundaries on abortion to extremes, the Republicans in Congress is rolling out a long-term push to force the Democrats to hold a vote on the bill.",1
"Microsoft has announced an ambitious effort to make voting secure, verifiable and subject to reliable audits by registering ballots in encrypted form so they can be accurately and independently tracked long after they are cast.",0
Pence was the source of inspiration to pro-life activists for a second time Friday as he had addressed the March for Life rally earlier in the day when he made a surprise visit to speak to thousands of pro-life marchers participating in the annual event.,0
The new numbers from Gallup are an unwelcome sight for Democrats after kicking off the week with a disaster caucus in Iowa who and simultaneously anticipating a Trump acquittal in the Senate. Trump will also now have the opportunity to shine in his newfound approval in Tuesday night’s address to the nation while Democrats are in disarray.,1
"Women’s rights advocates were relieved late Monday when Senate Democrats blocked an extreme anti-choice bill from advancing to a floor debate—but were soon outraged by President Donald Trump’s lies about abortion care, as well as the corporate media for enabling his blatantly false characterization of the bill to stand.",1
"Paul Whelan, the former U.S. Marine held in Moscow on spying charges, had online contact with more than 20 Russians with military backgrounds, an analysis of social media shows.",0
"But last Wednesday, when the Heritage Clinic for Women in Grand Rapids, Michigan, opened in the morning, the staff were startled to find 25 to 30 protesters assembling, many of them holding single red roses. Ignoring the no-trespassing signs, they began swarming the clinic’s parking lot, rushing patients as they got out of their cars.",1
"Professional camaraderie between fighters of different races in the sport of boxing is evidence of America moving beyond racial divisions, assessed Randy Gordon, host of SiriusXM’s At the Fights.",0
"Hungarian Prime Minister Viktor Orban, a staunch anti-immigrant populist, said on Wednesday a new wave of migrants trying to cross the border from Turkey into the European Union must be stopped as far south as possible and his government was ready to help frontline Greece.",1
"The House Democrats’ 1,400-page coronavirus recovery bill threatens the livelihood of millions of American graduates and their families by expanding work visas for many of the roughly 1.5 million foreign college graduate contract workers who hold jobs in the United States.",1
"My point is that it was a bait-and-switch, and not in a crazed, left-wing state like California. ",1
"The New York Times recently published a fairly detailed fact-check of these Trump World arguments that democrats don’t mind executing babies after birth, and not surprisingly, the rhetoric is irresponsibly wrong.",1
"A narrow plurality of Democrats favor allowing men who say they are transgender to enter women’s sports, but the change is opposed two-to-one by adults, independents, parents, and middle-class voters, says a poll by Rasmussen Reports.",0
"Ever since Democratic presidential candidate Sen. Elizabeth Warren (D-Mass.) revealed her plan to forgive student debt and make public universities free on Monday morning, the internet has been a carnival of bad faith, magical thinking and misinformation about the nature of college costs in the United States.",1
"As shown below, women are harmed by Planned Parenthood, and the vulnerable––such as sex abuse victims and the unborn––are sacrificed for profit and political advance.",1
"The coronavirus lockdowns are America’s most regressive government actions since the draft. Even as liberals have quickly noted the virus’s disparate incidence, they have ignored the inequities of government responses.",1
"I'm also struck by Blackburn's style of needlessly toxic politics, once again questioning the patriotism of a decorated U.S. Army combat veteran who earned a Purple Heart.",1
"There is no way to fling the footage into the Trash folder, or to edit the crap out of it until it sends the “right” message. Trump’s continued demands that the world act like a reality-show producer who can keep giving him mulligans only reaffirms that he is, above all things, a narcissistic moron.",1
"Schlapp’s apology comes as the U.S. is convulsed by protests after the police killing in Minnesota last month of a black man, George Floyd, by a white police officer, who knelt on his neck for nearly nine minutes. That officer has been charged with second degree murder and three fellow officers have been charged in abetting Floyd’s death.",1
"Britain will ban the sale of new petrol, diesel and hybrid cars from 2035, five years earlier than planned, in an attempt to reduce air pollution that could herald the end of over a century of reliance on the internal combustion engine.",0
Proponents of stricter protections for students argue the DOE’s recent rollback of Obama-era rules aimed at protecting students from predatory for-profit colleges will only lead to even more student failing to pay back their loans.,1
"On one hand, naive teenagers who signed off their financial futures to leftist, anti-American institutions would catch a break. The college cartel screwed you; now here’s a government waiver to make you forever grateful to the Democrats. ",1
A majority of Americans do not agree with Democrats’ gun violence demagoguery. ,1
"In his failed bid for the Democrat nomination for 2020, Beto O’Rourke said churches that oppose same-sex marriage should lose their tax-exempt status.  In other words, toe the non-Biblical party line or else!",1
"The plan would particularly benefit black, Latino and lower-income households, as well as households headed by people who never finished college, the researchers said.",0
"President Donald Trump offered an unusual warning to Virginia farmers on Tuesday, suggesting that gun control will leave their potatoes defenseless. ",1
"In the months that followed, all independent polling has found the American mainstream not only blames Trump and his party for the shutdown, but also does not want to spend billions of taxpayer dollars on an ineffective and unnecessary border wall.",1
Many Latino voters know from personal and family experience how U.S. foreign policies drive people from their homes. ,1
The transgender effort to suppress any recognition that men and women are different and complementary would not matter except for the movement’s political alliance with wealthy progressives and radical feminists who wish to destroy the political power of the male-and-female family.,1
"Since the migration crisis erupted in 2015, mainly fuelled by the war in Syria, Greece has granted asylum to around 40,000 people, Mitarachi said.",2
"U.S. President Donald Trump said on Tuesday he backed Bolivia’s interim President Jeanine Anez as she seeks “a peaceful democratic transition,” and he denounced ongoing violence in the country.",0
"Procter and Gamble, a major sponsor of the US Women’s National Soccer team, has backed the players in their fight for equal pay, with a public donation of $529,000.",0
"While health experts say a vaccine to prevent infection is needed to return life to normal, the survey points to a potential trust issue for the Trump administration already under fire for its often contradictory safety guidance during the pandemic.",1
"Israel has a vested interest in escalation, not de-escalation, with Iran.",1
"Brandon Straka, founder of the #WalkAway campaign encouraging Americans to leave the Democrat Party and left-wing ideology, told Breitbart News on Saturday that many participants in the Women’s March in Washington, DC, don’t really know why they’re here.",0
"That has led to fears that gangs could exploit undocumented migrants already in the country for cheap labor, an established but illegal practice known as “caporalato” that one United Nations expert has called a form of modern slavery.",1
Transgender MMA fighter Fallon Fox twice broke a female opponent’s skull to win matches and now he is being praised by some in the LGBT community.,1
"She’s among four dozen or so people gathered outside on a hot late-summer morning, joining a hard core of activists who believe that all vaccines are dangerous and who have become increasingly emboldened about denouncing the medical establishment.",1
"Moreover, NeverTrump would not only have to separate Trump from his political base, but also to win the support of that base for its own candidate. That will be particularly difficult, as NeverTrump has made no secret of its general disdain for Trump supporters.",1
"In February, Mexican illegal alien Pablo Hernandez was driving a pick-up truck when he struck a six-year-old boy and a 16-year-old teenage boy as they were crossing the street. ",2
U.S. President Donald Trump landed in London on Monday for a NATO summit. Queen Elizabeth will host the NATO leaders at Buckingham Palace on Tuesday.,0
The Brazilian superstar’s career is a testament to what’s possible in women’s soccer — and also a reminder of how little attention the sport still receives.,1
"Alabama has the most restrictive abortion law in the U.S., banning abortion at any stage of pregnancy and for any reason, including in cases of rape and incest.",0
Closing the gender pay gap is not rocket science – even though recently graduated female rocket scientists earn 89 cents on the dollar to their male peers.,1
"The anti-vaccine crowd frequently falls back on the claim that vaccines cause autism, despite more than two dozen studies clearly showing that vaccination does not increase the incidence of autism. Nor did it make a whit of biological sense that vaccines would.",1
"While Pace receives her spiritual grace from a Christian God, science has shown that faith in a supernatural wonder, whatever it may be, that bears powerful benefits on an individual’s mental and even physical well-being.",1
DeSantis was initially tentative to shut down outdoor recreational areas like beaches and was ridiculed by the media for his allegedly relaxed response.,1
"Rep. Alexandria Ocasio-Cortez, D-N.Y., has taken another shot at President Trump amid their ongoing feud, saying he cannot conceive of an immigration system that didn't involve torture or hurting innocent people.",2
The player vote ending Thursday at midnight—a simple majority vote will determine the outcome of the NFL CBA—is a monumental referendum for the next decade of the NFL.,0
"Ocasio-Cortez, who advocates for democratic socialism, is not the only Democrat in Washington, D.C., honing in on billionaires these days.",1 """
    print(repr(string[435:455]), sep='\n')

def create_big_dataset():
    """
    Helper function to create a big dataset from the smaller ones and concatenate them into one.
    """

    file_names = ("final_labels_MBIC_new.csv", "news_headlines_usa_biased.csv", "news_headlines_usa_neutral.csv", "raw_labels_SG2.csv")


    with open(f"{DATA_FOLDER_PATH}\\big_ds.csv", "w", encoding="utf-8") as out_csv_file:
        csv_writer = csv.writer(out_csv_file, delimiter=',', lineterminator="\n")
        
        for idx, file_name in enumerate(file_names):
            print(f"Starting {file_name}")

            if idx < 3:

                with open(f"{DATA_FOLDER_PATH}\\{file_name}", "r", encoding="latin") as in_csv_file:
                    csv_reader = csv.reader(in_csv_file)
                    if idx == 0:
                        file_iter = iter(csv_reader)
                        next(file_iter)
                        for row in file_iter:
                            csv_writer.writerow([row[0], row[7]])

                    elif idx == 1:
                        file_iter = iter(csv_reader)
                        next(file_iter)
                        for row in file_iter:
                            csv_writer.writerow((row[2], "Biased"))

                    elif idx == 2:
                        file_iter = iter(csv_reader)
                        next(file_iter)
                        for row in file_iter:
                            csv_writer.writerow((row[2], "Non-biased"))

            elif idx == 3:
                with open(f"{DATA_FOLDER_PATH}\\{file_name}", "r", encoding="utf-8") as in_csv_file:
                    csv_reader = csv.reader(in_csv_file, delimiter=';')
                    file_iter = iter(csv_reader)
                    next(file_iter)
                    for row in file_iter:
                        csv_writer.writerow((row[0], row[5]))

def big_ds_filter(threshold = 180):
    """
    A function that takes filters a dataset to only have rows with a length of less than or equal to the threshold.
    """
    with open(f"{BIG_DS_PATH}\\big_ds_cleaned.csv", 'r', encoding="utf-8") as in_csv:
        csv_reader = csv.reader(in_csv)
        with open(f"{BIG_DS_PATH}\\big_ds_cleaned_filtered.csv", 'w', encoding="utf-8") as out_csv:
            csv_writer = csv.writer(out_csv, delimiter=',', lineterminator="\n")
            for row in csv_reader:
                if len(row[0]) <= threshold:
                    csv_writer.writerow(row)



@execute_this
def main():
    big_ds_filter()
