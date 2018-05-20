SELECT SUM ( T2.* ) from Albums as T1 join Vocals as T2 WHERE T1.Label = UNKNOWN_VALUE	music_2
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	music_2
SELECT id from languages WHERE name in UNKNOWN_VALUE	music_2
SELECT Press_ID from press	music_2
SELECT T2.Instrument from Songs as T1 join Instruments as T2 on T1.SongId = T2.SongId join Vocals as T3 WHERE T1.Title = UNKNOWN_VALUE ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title like UNKNOWN_VALUE	music_2
SELECT T1.Country from country as T1 join team as T2 WHERE T2.Team_ID = UNKNOWN_VALUE	music_2
SELECT T1.dept_name from department as T1 join time_slot as T2 WHERE T2.day > UNKNOWN_VALUE	music_2
SELECT MIN ( health_score ) , MIN ( education_score ) from countries	tvshow
SELECT T2.name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name WHERE T1.dept_name > UNKNOWN_VALUE	music_2
SELECT Type from Albums	music_2
SELECT SUM ( T2.* ) from Products as T1 join Shipment_Items as T2 WHERE T1.product_description = UNKNOWN_VALUE	music_2
SELECT AVG ( shipment_id ) from Shipments	music_2
SELECT T1.Headphone_ID , Headphone_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY T3.Neighborhood	tvshow
SELECT AVG ( T3.Name ) from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY T1.Price	music_2
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name = UNKNOWN_VALUE	music_2
SELECT MIN ( shipment_date ) , COUNT ( shipment_date ) from Shipments	tvshow
SELECT Shop_ID , Shop_Name from shop WHERE Open_Date = UNKNOWN_VALUE	music_2
SELECT T1.Model from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID join stock as T4 GROUP BY T3.Date_Opened ORDER BY COUNT ( T4.* ) LIMIT 1	music_2
SELECT organisation_type from Organisation_Types	music_2
SELECT dept_name , dept_name from department	tvshow
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	music_2
SELECT id from languages	music_2
SELECT Food from goods WHERE Price > UNKNOWN_VALUE	music_2
SELECT T1.organisation_id from Grants as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	music_2
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	music_2
SELECT origin , flno from flight WHERE distance = UNKNOWN_VALUE	music_2
SELECT MIN ( T3.* ) , T2.Height , T2.Date_of_Birth from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID join people as T3 WHERE T1.People_ID = UNKNOWN_VALUE	music_2
SELECT MIN ( T2.country ) , MIN ( country ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join results as T3 on T1.raceId = T3.raceId GROUP BY T1.url ORDER BY T3.grid LIMIT 1	music_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	music_2
SELECT product_type_code from Products	music_2
SELECT SUM ( T2.* ) from results as T1 join lapTimes as T2 WHERE T1.fastestLapTime > UNKNOWN_VALUE	music_2
SELECT T2.* , T1.health_score from countries as T1 join official_languages as T2 GROUP BY T1.politics_score	tvshow
SELECT Shop_ID from shop WHERE Shop_ID = UNKNOWN_VALUE	music_2
SELECT T1.Version_Number from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T2.Document_ID ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT SUM ( T2.* ) from Products as T1 join Shipment_Items as T2 WHERE T1.product_description = UNKNOWN_VALUE	music_2
SELECT T1.Origin from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T3.Channel_ID = UNKNOWN_VALUE	music_2
SELECT T4.gender_code , T1.parent_product_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id	tvshow
SELECT MIN ( T4.email_address ) from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description > UNKNOWN_VALUE	music_2
SELECT T2.* from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	music_2
SELECT AVG ( T1.dept_name ) , AVG ( dept_name ) from department as T1 join time_slot as T2 GROUP BY T2.end_hr	tvshow
SELECT series_name from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	music_2
SELECT Shop_ID from shop WHERE Open_Date = UNKNOWN_VALUE ORDER BY LIMIT 1	music_2
SELECT SUM ( T1.Id ) , Id from customers as T1 join items as T2 GROUP BY T2.Item	tvshow
SELECT SUM ( T2.* ) , T1.outcome_details from Project_Outcomes as T1 join Tasks as T2 GROUP BY T1.project_id	tvshow
SELECT T1.Id from customers as T1 join receipts as T2 on T1.Id = T2.CustomerId WHERE T2.Date = UNKNOWN_VALUE	music_2
SELECT T1.customer_address , T1.customer_phone , T2.order_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id WHERE T1.payment_method_code = UNKNOWN_VALUE	music_2
SELECT AVG ( T2.* ) , T1.dept_name , dept_name from department as T1 join prereq as T2 join time_slot as T3 GROUP BY T3.end_min	country_language
SELECT SUM ( * ) from Claims_Processing	music_2
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code WHERE T1.Template_Type_Description = UNKNOWN_VALUE	music_2
SELECT AVG ( T3.Share_in_percent ) , Share_in_percent from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID GROUP BY T1.Program_ID	tvshow
SELECT SUM ( * ) from Vocals	music_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	music_2
SELECT AVG ( * ) , SUM ( * ) from Claims_Processing	tvshow
SELECT T1.series_name from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel join Cartoon as T3 GROUP BY T2.Viewers_m HAVING T3.*	music_2
SELECT T1.organisation_type from Organisation_Types as T1 join Organisations as T2 on T1.organisation_type = T2.organisation_type join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Staff as T4 on T3.project_id = T4.project_id WHERE T4.staff_id = UNKNOWN_VALUE	music_2
SELECT T1.Template_ID , T2.Document_ID , T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID	country_language
SELECT Id , Id from customers ORDER BY	tvshow
SELECT T1.Template_ID from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID WHERE T2.Document_Description like UNKNOWN_VALUE	music_2
SELECT AVG ( Owner ) from program	music_2
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID join Claims_Processing as T4 GROUP BY T3.Date_of_Claim ORDER BY COUNT ( T4.* ) LIMIT 1	music_2
SELECT T2.* , T1.Name from MovieTheaters as T1 join MovieTheaters as T2 GROUP BY T1.Movie ORDER BY COUNT ( * ) LIMIT 1	music_2
SELECT SUM ( * ) from stock	music_2
SELECT SUM ( T3.* ) , T1.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Template_Details	tvshow
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId WHERE T2.StagePosition = UNKNOWN_VALUE	music_2
SELECT SUM ( * ) from broadcast_share	music_2
SELECT T1.customer_phone from Customers as T1 join Order_Items as T2 GROUP BY T1.customer_number HAVING T2.*	music_2
SELECT T1.Position from Tracklists as T1 join Vocals as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	music_2
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	music_2
SELECT T2.Food from customers as T1 join goods as T2 join receipts as T3 WHERE T1.FirstName = UNKNOWN_VALUE ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT destination from flight	music_2
SELECT T1.customer_phone from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id join Order_Items as T3 on T2.order_id = T3.order_id join Order_Items as T4 GROUP BY T3.order_quantity ORDER BY COUNT ( T4.* ) LIMIT 1	music_2
SELECT T1.destination , destination from flight as T1 join employee as T2 ORDER BY T2.eid	music_2
SELECT Customer_Details from Customers	music_2
SELECT T2.* , T1.dept_name from department as T1 join prereq as T2	program_share
SELECT aid from flight	music_2
SELECT Version_Number from Templates	music_2
SELECT T2.* , T1.sec_id from section as T1 join prereq as T2 WHERE T1.semester < UNKNOWN_VALUE ORDER BY LIMIT 1	music_2
SELECT destination , flno from flight	program_share
SELECT SUM ( T2.* ) from Grants as T1 join Tasks as T2 WHERE T1.grant_end_date > UNKNOWN_VALUE	music_2
SELECT T3.organisation_id from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id WHERE T1.document_description = UNKNOWN_VALUE	music_2
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	program_share
SELECT AVG ( T1.Template_Type_Code ) , SUM ( T3.* ) from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Template_ID	program_share
SELECT T2.* , T1.Car_# from driver as T1 join team_driver as T2	program_share
SELECT T2.* from Movies as T1 join MovieTheaters as T2 ORDER BY T1.Title	music_2
SELECT T3.Press_ID from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE T1.Age in UNKNOWN_VALUE	music_2
SELECT T3.* , T2.country from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T1.raceId = UNKNOWN_VALUE	music_2
SELECT SUM ( T3.* ) , SUM ( * ) , T2.circuitRef from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.raceId HAVING *	customers_and_orders
SELECT T1.Template_ID , T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID WHERE T1.Template_Details = UNKNOWN_VALUE	music_2
SELECT T3.* , T2.order_item_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Shipment_Items as T3 WHERE T1.product_description = UNKNOWN_VALUE	music_2
SELECT id from languages WHERE id in UNKNOWN_VALUE	music_2
SELECT SUM ( flno ) from flight	music_2
SELECT Id from customers	music_2
SELECT Channel_ID from channel	music_2
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id WHERE T3.justice_score like UNKNOWN_VALUE	music_2
SELECT Store_ID from store ORDER BY Store_ID	music_2
SELECT LastName from customers	music_2
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE GROUP BY T3.order_id HAVING T1.product_id	music_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	music_2
SELECT surname , dob from drivers	program_share
SELECT SUM ( * ) from Vocals	music_2
SELECT T2.Nov from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.City = UNKNOWN_VALUE	music_2
SELECT AVG ( dept_name ) , AVG ( dept_name ) from department ORDER BY LIMIT 1	program_share
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T1.Template_Type_Code ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT T3.organisation_type from Research_Staff as T1 join Organisations as T2 on T1.employer_organisation_id = T2.organisation_id join Organisation_Types as T3 on T2.organisation_type = T3.organisation_type WHERE T1.staff_details = UNKNOWN_VALUE	music_2
SELECT SUM ( T1.Book_Series ) , T2.* from book as T1 join book as T2 GROUP BY T1.Release_date	program_share
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.aid = UNKNOWN_VALUE	music_2
SELECT SUM ( T1.aid ) , T2.* from flight as T1 join certificate as T2 GROUP BY T1.origin	program_share
SELECT T5.organisation_type from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type WHERE T1.document_type_code = UNKNOWN_VALUE	music_2
SELECT FirstName from customers	music_2
SELECT T1.dept_name , T2.name , T5.semester from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id join prereq as T6 GROUP BY T5.year HAVING T6.* ORDER BY LIMIT 1	customers_and_orders
SELECT T2.Version_Number , T1.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code	program_share
SELECT T1.Code from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie GROUP BY T2.Movie HAVING Code	music_2
SELECT T1.dept_name , dept_name from department as T1 join time_slot as T2 ORDER BY T2.end_hr	music_2
SELECT T2.customer_phone from Addresses as T1 join Customers as T2 join Order_Items as T3 GROUP BY T1.address_id ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT T2.* , T1.justice_score from countries as T1 join official_languages as T2 WHERE T1.economics_score = UNKNOWN_VALUE	music_2
SELECT T4.eg Agree Objectives from Grants as T1 join Organisations as T2 on T1.organisation_id = T2.organisation_id join Projects as T3 on T2.organisation_id = T3.organisation_id join Tasks as T4 on T3.project_id = T4.project_id WHERE T1.grant_end_date = UNKNOWN_VALUE	music_2
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV in UNKNOWN_VALUE	music_2
SELECT T1.Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Winnings ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT gender_code , customer_id from Customers	program_share
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE GROUP BY T4.year LIMIT 1	music_2
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id WHERE T3.economics_score > UNKNOWN_VALUE	music_2
SELECT T1.Unsure_rate from candidate as T1 join people as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	music_2
SELECT AVG ( T2.Date ) , Date from channel as T1 join broadcast_share as T2 on T1.Channel_ID = T2.Channel_ID GROUP BY T1.Channel_ID	program_share
SELECT Origin from program WHERE Launch like UNKNOWN_VALUE	music_2
SELECT T1.Customer_ID from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID ORDER BY T3.Claim_Header_ID	music_2
SELECT Origin from program ORDER BY Origin	music_2
SELECT Capital from country	music_2
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID ORDER BY T1.Software_Platform	device
SELECT T2.salary , T1.name from aircraft as T1 join employee as T2	flight_1
SELECT SUM ( Id ) from customers WHERE LastName in UNKNOWN_VALUE	device
SELECT T1.Version_Number from Templates as T1 join Paragraphs as T2 GROUP BY T1.Template_ID ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT MIN ( T2.country ) , MIN ( T2.lat ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join results as T3 on T1.raceId = T3.raceId WHERE T3.fastestLapSpeed > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T1.year	device
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID GROUP BY T1.Template_Type_Code HAVING T3.Document_Name	device
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	device
SELECT AVG ( T5.dept_name ) , SUM ( T1.building ) from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name	flight_1
SELECT SUM ( * ) from Shipment_Items	device
SELECT Code from Movies	device
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY T1.Earpads HAVING T1.Headphone_ID	device
SELECT AVG ( T3.eid ) , T2.name from flight as T1 join aircraft as T2 join employee as T3 GROUP BY T1.departure_date	flight_1
SELECT T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE GROUP BY T5.year ORDER BY SUM ( T2.ID ) LIMIT 1	device
SELECT SUM ( T2.* ) from author as T1 join book as T2 WHERE T1.Name > UNKNOWN_VALUE	device
SELECT T1.Franchise from game as T1 join game_player as T2 on T1.Game_ID = T2.Game_ID join player as T3 on T2.Player_ID = T3.Player_ID WHERE T3.Player_ID = UNKNOWN_VALUE	device
SELECT T1.Press_ID , T1.Year_Profits_billion , Year_Profits_billion from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID ORDER BY T2.Book_ID LIMIT 1	device
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	device
SELECT T1.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE GROUP BY T5.year ORDER BY T2.ID LIMIT 1	device
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T2.Bandmate ORDER BY COUNT ( T3.* ) LIMIT 1	device
SELECT T1.organisation_id from Grants as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT aid from flight WHERE aid = UNKNOWN_VALUE	device
SELECT SUM ( T2.* ) , T1.Document_Name from Documents as T1 join Paragraphs as T2 GROUP BY Document_Name ORDER BY COUNT ( * )	device
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id GROUP BY T2.language_id HAVING id	device
SELECT T1.People_ID , T2.Name from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID ORDER BY T2.Sex LIMIT 1	device
SELECT T2.alt , alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T3.*	device
SELECT T1.project_id from Project_Outcomes as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT MIN ( T4.email_address ) , COUNT ( T1.product_description ) from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	device
SELECT Instrument from Instruments	device
SELECT SUM ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	device
SELECT T1.Game_ID from game as T1 join game_player as T2 on T1.Game_ID = T2.Game_ID join player as T3 on T2.Player_ID = T3.Player_ID join game_player as T4 GROUP BY T3.Rank_of_the_year ORDER BY COUNT ( T4.* ) LIMIT 1	device
SELECT Id , Id from customers GROUP BY Id	flight_1
SELECT Food from goods	device
SELECT T4.customer_address , T4.customer_phone , T3.order_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T1.product_price ORDER BY COUNT ( T5.* ) LIMIT 1	device
SELECT T1.dept_name , dept_name from department as T1 join prereq as T2 join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID GROUP BY T4.grade HAVING T2.*	flight_1
SELECT SUM ( T4.alt ) , T5.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join circuits as T4 on T1.circuitId = T4.circuitId join lapTimes as T5 WHERE T4.circuitRef > UNKNOWN_VALUE GROUP BY T1.url ORDER BY T3.driverId LIMIT 1	device
SELECT City from city WHERE Hanzi = UNKNOWN_VALUE	device
SELECT Id from customers ORDER BY LastName	device
SELECT T1.Store_ID from store as T1 join stock as T2 GROUP BY T1.Date_Opened ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID ORDER BY T1.Price LIMIT 1	device
SELECT Document_Description from Documents	device
SELECT T2.name , T1.aid from flight as T1 join aircraft as T2	flight_1
SELECT Shop_ID from shop WHERE Open_Date > UNKNOWN_VALUE	device
SELECT destination from flight	device
SELECT T5.shipment_id from Products as T1 join Shipment_Items as T2 join Order_Items as T3 on T1.product_id = T3.product_id join Shipment_Items as T4 on T3.order_item_id = T4.order_item_id join Shipments as T5 on T4.shipment_id = T5.shipment_id WHERE T1.product_description = UNKNOWN_VALUE GROUP BY T5.shipment_date ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT id from languages ORDER BY name	device
SELECT T1.product_id from Products as T1 join Order_Items as T2 GROUP BY T1.product_type_code ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT * from MovieTheaters	device
SELECT destination from flight	device
SELECT Position from Tracklists	device
SELECT T3.dept_name , dept_name from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join department as T3 on T2.dept_name = T3.dept_name GROUP BY T1.i_ID LIMIT 1	flight_1
SELECT SUM ( * ) from Vocals	device
SELECT T1.Shop_ID from shop as T1 join stock as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT SUM ( * ) from Order_Items	device
SELECT MIN ( T3.Document_Name ) , MIN ( Document_Name ) from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID WHERE T1.Template_Type_Description > UNKNOWN_VALUE	device
SELECT T2.Game_ID , T1.Market_district from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID ORDER BY Market_district	device
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	device
SELECT dept_name , dept_name from department ORDER BY LIMIT 1	flight_1
SELECT SUM ( * ) from MovieTheaters	device
SELECT SUM ( * ) from Paragraphs	device
SELECT T1.Platform_name from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID join game_player as T5 GROUP BY T4.Player_ID ORDER BY COUNT ( T5.* ) LIMIT 1	device
SELECT T1.dept_name , dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year ORDER BY LIMIT 1	bakery_1
SELECT T3.Other_Details from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID WHERE T1.Template_Type_Code = UNKNOWN_VALUE	device
SELECT City from city WHERE City > UNKNOWN_VALUE	device
SELECT T2.project_details from Project_Outcomes as T1 join Projects as T2 on T1.project_id = T2.project_id WHERE T1.project_id = UNKNOWN_VALUE	device
SELECT T3.Store_ID , T1.Headphone_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Class = UNKNOWN_VALUE	headphone_store
SELECT T1.gender_code , T1.customer_last_name from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipment_Items as T3 WHERE T3.* = UNKNOWN_VALUE GROUP BY T2.order_id HAVING *	headphone_store
SELECT SUM ( Id ) from customers	headphone_store
SELECT MIN ( Date_of_Birth ) , MIN ( Date_of_Birth ) from people	e_commerce
SELECT dept_name from department	headphone_store
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Class in UNKNOWN_VALUE	headphone_store
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 GROUP BY T3.politics_score ORDER BY COUNT ( T4.* ) LIMIT 1	headphone_store
SELECT SUM ( T1.driverRef ) , T2.* from drivers as T1 join lapTimes as T2 join qualifying as T3 on T1.driverId = T3.driverId GROUP BY T3.q1	e_commerce
SELECT Hight_definition_TV from TV_Channel	headphone_store
SELECT People_ID from candidate GROUP BY Oppose_rate	headphone_store
SELECT SUM ( * ) from Order_Items	headphone_store
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id	e_commerce
SELECT * from MovieTheaters	headphone_store
SELECT MIN ( T1.email_address ) from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipments as T3 on T2.order_id = T3.order_id join Invoices as T4 on T3.invoice_number = T4.invoice_number join Shipment_Items as T5 WHERE T5.* > UNKNOWN_VALUE GROUP BY T4.invoice_status_code HAVING *	headphone_store
SELECT origin from flight WHERE distance = UNKNOWN_VALUE	headphone_store
SELECT T1.customer_address , T2.order_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id	e_commerce
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	headphone_store
SELECT T1.Origin from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID ORDER BY T3.Rating_in_percent LIMIT 1	headphone_store
SELECT Id from customers	headphone_store
SELECT MIN ( T1.Food ) , MIN ( Food ) from goods as T1 join items as T2 on T1.Id = T2.Item GROUP BY T2.Ordinal	e_commerce
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	headphone_store
SELECT T4.Oct from match as T1 join hosting_city as T2 on T1.Match_ID = T2.Match_ID join city as T3 on T2.Host_City = T3.City_ID join temperature as T4 on T3.City_ID = T4.City_ID WHERE T4.Nov = UNKNOWN_VALUE ORDER BY T1.Result LIMIT 1	headphone_store
SELECT series_name , Hight_definition_TV from TV_Channel	e_commerce
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Food > UNKNOWN_VALUE	headphone_store
SELECT SUM ( T3.driverRef ) , T4.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join lapTimes as T4 GROUP BY T1.raceId HAVING *	e_commerce
SELECT SUM ( * ) from game_player	headphone_store
SELECT T2.name from time_slot as T1 join instructor as T2 WHERE T1.start_hr > UNKNOWN_VALUE	headphone_store
SELECT SUM ( T2.* ) , * , * from Products as T1 join Order_Items as T2 GROUP BY T1.product_type_code	bakery_1
SELECT dept_name from department ORDER BY LIMIT 1	headphone_store
SELECT T1.Title from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T2.Position ORDER BY COUNT ( T3.* ) LIMIT 1	headphone_store
SELECT SUM ( T2.* ) from Customers as T1 join Shipment_Items as T2 WHERE * = UNKNOWN_VALUE GROUP BY T1.country	headphone_store
SELECT T1.Hight_definition_TV from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel GROUP BY T2.Weekly_Rank ORDER BY T2.Episode	headphone_store
SELECT dept_name , dept_name from department	e_commerce
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.LastName in UNKNOWN_VALUE	headphone_store
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID GROUP BY T2.Name ORDER BY Name LIMIT 1	headphone_store
SELECT AVG ( aid ) from flight	headphone_store
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year HAVING T3.course_id	e_commerce
SELECT Country from country	headphone_store
SELECT Origin from program WHERE Owner = UNKNOWN_VALUE	headphone_store
SELECT destination from flight	headphone_store
SELECT T1.dept_name , T4.sec_id , T4.semester from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id ORDER BY LIMIT 1	bakery_1
SELECT SUM ( * ) from Order_Items	headphone_store
SELECT Press_ID from press	headphone_store
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id	e_commerce
SELECT SUM ( * ) from Tasks	headphone_store
SELECT Shop_ID from shop WHERE Shop_Name = UNKNOWN_VALUE	headphone_store
SELECT T3.Document_ID from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID join Paragraphs as T4 GROUP BY T1.Template_Type_Description ORDER BY COUNT ( T4.* ) LIMIT 1	headphone_store
SELECT Id from customers WHERE LastName = UNKNOWN_VALUE	headphone_store
SELECT AVG ( T1.Next_Claim_Stage_ID ) , SUM ( T2.* ) from Claims_Processing_Stages as T1 join Claims_Processing as T2	e_commerce
SELECT Press_ID from press	headphone_store
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID join people as T3 GROUP BY T2.Sex ORDER BY COUNT ( T3.* ) LIMIT 1	headphone_store
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id	e_commerce
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device in UNKNOWN_VALUE	headphone_store
SELECT Food from goods	headphone_store
SELECT AVG ( T3.salary ) , T2.name from flight as T1 join aircraft as T2 join employee as T3 GROUP BY T1.departure_date	e_commerce
SELECT SUM ( T1.Id ) from customers as T1 join receipts as T2 on T1.Id = T2.CustomerId WHERE T1.FirstName = UNKNOWN_VALUE GROUP BY T2.ReceiptNumber	headphone_store
SELECT MIN ( Title ) , MIN ( Title ) from book	e_commerce
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Food > UNKNOWN_VALUE	headphone_store
SELECT Document_Description from Documents WHERE Document_ID = UNKNOWN_VALUE	headphone_store
SELECT product_type_code from Products ORDER BY product_type_code	headphone_store
SELECT T2.* from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	headphone_store
SELECT dept_name , dept_name , dept_name from department ORDER BY LIMIT 1	bakery_1
SELECT Id from customers	headphone_store
SELECT Store_ID from store WHERE Date_Opened = UNKNOWN_VALUE	headphone_store
SELECT MIN ( T2.Height ) , SUM ( T3.* ) from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID join people as T3 GROUP BY T1.People_ID	e_commerce
SELECT parent_product_id from Products WHERE product_description = UNKNOWN_VALUE	candidate_poll
SELECT Customer_Details from Customers	candidate_poll
SELECT SUM ( * ) from Tasks	candidate_poll
SELECT SUM ( * ) from Order_Items	candidate_poll
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	candidate_poll
SELECT Store_ID from store	candidate_poll
SELECT product_price from Products ORDER BY product_type_code	candidate_poll
SELECT SUM ( T2.* ) from Grants as T1 join Tasks as T2 WHERE T1.grant_end_date > UNKNOWN_VALUE	candidate_poll
SELECT MIN ( order_id ) , COUNT ( order_id ) from Customer_Orders	e_commerce
SELECT T4.customer_email , customer_email , T3.order_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T1.product_price ORDER BY COUNT ( T5.* ) LIMIT 1	candidate_poll
SELECT T1.gender_code , T1.customer_last_name from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipment_Items as T3 WHERE T3.* = UNKNOWN_VALUE GROUP BY T2.order_id HAVING *	candidate_poll
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	candidate_poll
SELECT T1.organisation_type from Organisation_Types as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	candidate_poll
SELECT SUM ( * ) from book	candidate_poll
SELECT T3.Type from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T1.SongId = UNKNOWN_VALUE	candidate_poll
SELECT Version_Number from Templates WHERE Template_Details = UNKNOWN_VALUE	candidate_poll
SELECT AVG ( T2.Document_Name ) from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID GROUP BY T1.Version_Number HAVING T2.Document_Description	candidate_poll
SELECT SUM ( Id ) , Id from customers	e_commerce
SELECT T1.Title from Songs as T1 join Vocals as T2 on T1.SongId = T2.SongId join Band as T3 on T2.Bandmate = T3.Id join Vocals as T4 GROUP BY T3.Lastname ORDER BY COUNT ( T4.* ) LIMIT 1	candidate_poll
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Driver-matched_dB = UNKNOWN_VALUE	candidate_poll
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY dept_name LIMIT 1	candidate_poll
SELECT T1.Id , Id , T2.Food from customers as T1 join goods as T2 ORDER BY Food	candidate_poll
SELECT SUM ( * ) from stock	candidate_poll
SELECT Code from Movies WHERE Code = UNKNOWN_VALUE	candidate_poll
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Flavor > UNKNOWN_VALUE	candidate_poll
SELECT T1.origin , T1.destination from flight as T1 join certificate as T2 GROUP BY T1.aid ORDER BY COUNT ( T2.* ) LIMIT 1	candidate_poll
SELECT T1.number , T4.country from drivers as T1 join lapTimes as T2 on T1.driverId = T2.driverId join races as T3 on T2.raceId = T3.raceId join circuits as T4 on T3.circuitId = T4.circuitId join results as T5 on T1.driverId = T5.driverId WHERE T5.fastestLapTime > UNKNOWN_VALUE	candidate_poll
SELECT T2.* from Movies as T1 join MovieTheaters as T2 WHERE T1.Rating = UNKNOWN_VALUE	candidate_poll
SELECT MIN ( * ) from Shipment_Items WHERE * = UNKNOWN_VALUE	candidate_poll
SELECT dept_name , dept_name from department	e_commerce
SELECT T1.product_type_code from Products as T1 join Order_Items as T2 GROUP BY T1.product_id ORDER BY COUNT ( T2.* ) LIMIT 1	candidate_poll
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE GROUP BY T4.customer_id HAVING *	candidate_poll
SELECT T2.Title , T1.Market_district from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID WHERE T4.Player_ID = UNKNOWN_VALUE	candidate_poll
SELECT SUM ( T3.* ) , T2.circuitRef from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.raceId HAVING *	e_commerce
SELECT T1.Id from customers as T1 join goods as T2 join receipts as T3 GROUP BY T2.Flavor ORDER BY COUNT ( T3.* ) LIMIT 1	candidate_poll
SELECT T3.Claim_Type_Code from Staff as T1 join Claims_Processing as T2 on T1.Staff_ID = T2.Staff_ID join Claim_Headers as T3 on T2.Claim_ID = T3.Claim_Header_ID WHERE T1.Staff_Details = UNKNOWN_VALUE	candidate_poll
SELECT T1.document_id from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Projects as T4 on T3.organisation_id = T4.organisation_id join Project_Outcomes as T5 on T4.project_id = T5.project_id join Tasks as T6 GROUP BY T5.project_id ORDER BY COUNT ( T6.* ) LIMIT 1	candidate_poll
SELECT SUM ( * ) from Claims_Processing	candidate_poll
SELECT Origin from program ORDER BY Program_ID LIMIT 1	candidate_poll
SELECT T1.Customer_Details from Customers as T1 join Claims_Processing as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	candidate_poll
SELECT SUM ( * ) from Tasks	candidate_poll
SELECT MIN ( T3.Year_Profits_billion ) , MIN ( Year_Profits_billion ) from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID GROUP BY T1.Gender	e_commerce
SELECT T5.organisation_type from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type WHERE T1.document_description = UNKNOWN_VALUE	candidate_poll
SELECT MIN ( T4.invoice_number ) from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipments as T3 on T2.order_id = T3.order_id join Invoices as T4 on T3.invoice_number = T4.invoice_number join Shipment_Items as T5 GROUP BY T1.customer_id ORDER BY COUNT ( T5.* ) LIMIT 1	candidate_poll
SELECT T4.customer_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description like UNKNOWN_VALUE	candidate_poll
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 GROUP BY T1.SongId	candidate_poll
SELECT Id from customers WHERE FirstName > UNKNOWN_VALUE	candidate_poll
SELECT T5.dept_name , dept_name from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name join course as T6 on T2.course_id = T6.course_id GROUP BY T6.course_id ORDER BY SUM ( T1.room_number ) LIMIT 1	candidate_poll
SELECT SUM ( T4.* ) from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID join book as T4 WHERE T1.Author_ID > UNKNOWN_VALUE GROUP BY T3.Press_ID	candidate_poll
SELECT T1.alt , alt from circuits as T1 join lapTimes as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	candidate_poll
SELECT T2.Claim_Type_Code from Policies as T1 join Claim_Headers as T2 on T1.Policy_ID = T2.Policy_ID GROUP BY T1.Start_Date HAVING T2.Policy_ID	candidate_poll
SELECT T1.product_price , T1.product_type_code from Products as T1 join Addresses as T2 WHERE T2.address_id = UNKNOWN_VALUE	candidate_poll
SELECT T1.dept_name , dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	candidate_poll
SELECT Id from customers ORDER BY LastName	candidate_poll
SELECT T1.Country , T2.Make from country as T1 join team as T2	e_commerce
SELECT T1.Id , T2.Food from customers as T1 join goods as T2 join receipts as T3 on T1.Id = T3.CustomerId WHERE T3.ReceiptNumber > UNKNOWN_VALUE	candidate_poll
SELECT T3.Channel_ID from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T1.Name = UNKNOWN_VALUE ORDER BY T3.Rating_in_percent	candidate_poll
SELECT T3.* , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 WHERE T2.Units_sold_Millions = UNKNOWN_VALUE	candidate_poll
SELECT customer_email , customer_id from Customers GROUP BY customer_address HAVING customer_name	e_commerce
SELECT Id from customers	candidate_poll
SELECT T2.name , T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE	candidate_poll
SELECT SUM ( * ) from Tasks ORDER BY COUNT ( * ) LIMIT 1	candidate_poll
SELECT T1.id from languages as T1 join official_languages as T2 ORDER BY COUNT ( T2.* )	candidate_poll
SELECT Id from customers	bakery_1
SELECT MIN ( Food ) , COUNT ( Food ) from goods GROUP BY Food	bakery_1
SELECT T1.gender_code , T1.customer_last_name from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id GROUP BY T2.order_id	bakery_1
SELECT SUM ( T2.* ) , T1.Channel_ID from channel as T1 join broadcast_share as T2	bakery_1
SELECT flno , flno , flno from flight	e_commerce
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	bakery_1
SELECT Id , Id from customers ORDER BY Id	bakery_1
SELECT T2.Title , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID WHERE T2.Units_sold_Millions = UNKNOWN_VALUE	bakery_1
SELECT SUM ( T1.id ) , T4.* from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 GROUP BY T3.politics_score	bakery_1
SELECT SUM ( * ) from stock	bakery_1
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year ORDER BY year LIMIT 1	bakery_1
SELECT SUM ( * ) from Claims_Processing	bakery_1
SELECT MIN ( T2.Food ) , MIN ( Food ) , MIN ( Food ) from customers as T1 join goods as T2 GROUP BY T1.Id	e_commerce
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device_ID > UNKNOWN_VALUE	bakery_1
SELECT Store_ID from store WHERE Name > UNKNOWN_VALUE	bakery_1
SELECT T1.Title from Songs as T1 join Vocals as T2 GROUP BY T1.SongId ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT T1.Shop_ID from shop as T1 join stock as T2 ORDER BY T2.*	bakery_1
SELECT origin , flno , flno from flight	e_commerce
SELECT Country from country	bakery_1
SELECT T1.destination from flight as T1 join employee as T2 WHERE T2.eid = UNKNOWN_VALUE	bakery_1
SELECT Country from country	bakery_1
SELECT Title from Movies WHERE Code = UNKNOWN_VALUE	bakery_1
SELECT product_price from Products	bakery_1
SELECT SUM ( T2.alt ) , T3.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T1.raceId = UNKNOWN_VALUE	bakery_1
SELECT SUM ( T2.* ) from MovieTheaters as T1 join MovieTheaters as T2 GROUP BY T1.Name	bakery_1
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year ORDER BY COUNT ( T5.* ) LIMIT 1	bakery_1
SELECT Hight_definition_TV from TV_Channel ORDER BY id	bakery_1
SELECT MIN ( health_score ) , MIN ( health_score ) from countries WHERE health_score in UNKNOWN_VALUE	bakery_1
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	bakery_1
SELECT T1.Store_ID from store as T1 join stock as T2 GROUP BY T1.Name ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT People_ID from candidate WHERE Unsure_rate < UNKNOWN_VALUE	bakery_1
SELECT Template_Type_Code , Template_Type_Code from Ref_Template_Types	bakery_1
SELECT MIN ( T1.email_address ) , MIN ( T3.shipment_tracking_number ) from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipments as T3 on T2.order_id = T3.order_id join Invoices as T4 on T3.invoice_number = T4.invoice_number GROUP BY T4.invoice_status_code	bakery_1
SELECT Code from Movies WHERE Rating = UNKNOWN_VALUE	bakery_1
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	bakery_1
SELECT MIN ( T1.People_ID ) , MIN ( T2.Sex ) from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID ORDER BY People_ID	bakery_1
SELECT T1.aid , T1.destination from flight as T1 join certificate as T2 GROUP BY T1.price ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT county from Customers	bakery_1
SELECT organisation_type , organisation_type_description from Organisation_Types	bakery_1
SELECT T1.product_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Shipment_Items as T3 on T2.order_item_id = T3.order_item_id join Shipments as T4 on T3.shipment_id = T4.shipment_id join Invoices as T5 on T4.invoice_number = T5.invoice_number join Shipment_Items as T6 WHERE T6.* = UNKNOWN_VALUE GROUP BY T5.invoice_date ORDER BY COUNT ( * ) LIMIT 1	bakery_1
SELECT SUM ( T2.* ) from book as T1 join book as T2 GROUP BY T1.Book_ID	bakery_1
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code WHERE T1.Template_Type_Description = UNKNOWN_VALUE	bakery_1
SELECT T4.alt from drivers as T1 join lapTimes as T2 on T1.driverId = T2.driverId join races as T3 on T2.raceId = T3.raceId join circuits as T4 on T3.circuitId = T4.circuitId WHERE T1.driverId > UNKNOWN_VALUE	bakery_1
SELECT MIN ( product_description ) from Products WHERE product_description > UNKNOWN_VALUE	bakery_1
SELECT SUM ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	bakery_1
SELECT MIN ( T2.alt ) , MIN ( T2.lat ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T3.*	bakery_1
SELECT Title from Songs WHERE Title = UNKNOWN_VALUE	bakery_1
SELECT SUM ( * ) from Paragraphs	bakery_1
SELECT T1.gender_code from Customers as T1 join Shipment_Items as T2 WHERE T2.* > UNKNOWN_VALUE	bakery_1
SELECT T4.* , T3.Car_# from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID join team_driver as T4 WHERE T1.Team_ID > UNKNOWN_VALUE	bakery_1
SELECT Id , Id from customers	bakery_1
SELECT T1.Hight_definition_TV from TV_Channel as T1 join Cartoon as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT SUM ( * ) from Vocals	bakery_1
SELECT T1.Hight_definition_TV , T2.Channel from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel join Cartoon as T3 on T1.id = T3.Channel GROUP BY T3.Title ORDER BY T2.Episode LIMIT 1	bakery_1
SELECT Code , Code from Movies WHERE Code = UNKNOWN_VALUE	bakery_1
SELECT T1.Template_ID , T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 ORDER BY COUNT ( T3.* ) LIMIT 1	bakery_1
SELECT T2.* , T1.dept_name from department as T1 join prereq as T2 ORDER BY LIMIT 1	bakery_1
SELECT T4.gender_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description in UNKNOWN_VALUE	bakery_1
SELECT T2.* , T1.health_score from countries as T1 join official_languages as T2 GROUP BY T1.politics_score	bakery_1
SELECT T1.aid from flight as T1 join employee as T2 ORDER BY T2.eid LIMIT 1	bakery_1
SELECT Version_Number from Templates	bakery_1
SELECT T1.Id from customers as T1 join receipts as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT MIN ( T2.order_id ) , MIN ( order_id ) from Addresses as T1 join Customer_Orders as T2 GROUP BY T1.address_id	bakery_1
SELECT T4.* , T3.justice_score from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 WHERE T1.name = UNKNOWN_VALUE GROUP BY T3.politics_score	bakery_1
SELECT T1.name from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T1.year	formula_1
SELECT COUNT ( T2.* ) , T1.Platform_ID , Platform_ID from platform as T1 join game_player as T2	college_2
SELECT T1.dept_name from department as T1 join time_slot as T2 GROUP BY T2.end_hr LIMIT 1	formula_1
SELECT T2.alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId WHERE T1.time > UNKNOWN_VALUE GROUP BY T1.raceId	formula_1
SELECT T1.address_id , T2.order_id , order_id from Addresses as T1 join Customer_Orders as T2	college_2
SELECT SUM ( * ) from broadcast_share	formula_1
SELECT People_ID from candidate ORDER BY Date	formula_1
SELECT T1.Shop_ID from shop as T1 join stock as T2 ORDER BY T2.*	formula_1
SELECT destination from flight	formula_1
SELECT T3.Match_ID from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID WHERE T1.Hanzi > UNKNOWN_VALUE	formula_1
SELECT T2.* from city as T1 join hosting_city as T2 WHERE T1.City > UNKNOWN_VALUE	formula_1
SELECT dept_name , dept_name , dept_name from department ORDER BY LIMIT 1	college_2
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Price > UNKNOWN_VALUE	formula_1
SELECT name , name from aircraft	video_game
SELECT Hight_definition_TV from TV_Channel WHERE id = UNKNOWN_VALUE	formula_1
SELECT Title from game	formula_1
SELECT SUM ( T2.Document_ID ) , T3.* from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T1.Version_Number	video_game
SELECT T3.Type from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T1.Title like UNKNOWN_VALUE	formula_1
SELECT SUM ( * ) from Paragraphs	formula_1
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID join stock as T4 GROUP BY T1.Software_Platform ORDER BY COUNT ( T4.* ) LIMIT 1	formula_1
SELECT SUM ( T2.* ) from Addresses as T1 join Order_Items as T2 WHERE T1.address_id = UNKNOWN_VALUE	formula_1
SELECT Store_ID from store WHERE Neighborhood > UNKNOWN_VALUE	formula_1
SELECT T1.gender_code from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipments as T3 on T2.order_id = T3.order_id join Invoices as T4 on T3.invoice_number = T4.invoice_number join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE GROUP BY T4.invoice_status_code HAVING *	formula_1
SELECT SUM ( * ) from Tasks	formula_1
SELECT product_id from Products GROUP BY product_name	formula_1
SELECT SUM ( * ) from broadcast_share	formula_1
SELECT T2.Franchise , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID ORDER BY T1.Market_district	formula_1
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.LastName like UNKNOWN_VALUE	formula_1
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	formula_1
SELECT T1.Id , Id from customers as T1 join receipts as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	formula_1
SELECT T1.organisation_id from Grants as T1 join Organisations as T2 on T1.organisation_id = T2.organisation_id join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Staff as T4 on T3.project_id = T4.project_id WHERE T4.role_code = UNKNOWN_VALUE	formula_1
SELECT Document_Description , Document_Name , Document_Name from Documents	college_2
SELECT T1.Name , T3.Year_Profits_billion from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID ORDER BY T3.Press_ID	formula_1
SELECT SUM ( T3.Shop_ID ) from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID GROUP BY T1.Applications	formula_1
SELECT SUM ( T4.customer_phone ) , T5.* from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T1.product_type_code	video_game
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year HAVING T4.semester	video_game
SELECT T2.Nov from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.City = UNKNOWN_VALUE	formula_1
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	formula_1
SELECT T2.* from team as T1 join team_driver as T2 WHERE T1.Team_ID = UNKNOWN_VALUE	formula_1
SELECT T1.origin , T1.flno from flight as T1 join aircraft as T2 GROUP BY T2.name ORDER BY name	formula_1
SELECT T4.Dec from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Result LIMIT 1	formula_1
SELECT SUM ( T2.Other_Details ) , T3.* from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T1.Version_Number	video_game
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	formula_1
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join pitStops as T3 on T2.raceId = T3.raceId WHERE T3.lap > UNKNOWN_VALUE	formula_1
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	formula_1
SELECT Unsure_rate from candidate WHERE People_ID < UNKNOWN_VALUE	formula_1
SELECT T1.Hight_definition_TV from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel join Cartoon as T3 on T1.id = T3.Channel WHERE T3.Directed_by = UNKNOWN_VALUE ORDER BY T2.Channel LIMIT 1	formula_1
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	formula_1
SELECT T5.organisation_type , T6.outcome_details from Staff_Roles as T1 join Project_Staff as T2 on T1.role_code = T2.role_code join Projects as T3 on T2.project_id = T3.project_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type join Project_Outcomes as T6 on T3.project_id = T6.project_id WHERE T1.role_description = UNKNOWN_VALUE	formula_1
SELECT T4.gender_code , T4.customer_last_name from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description = UNKNOWN_VALUE	formula_1
SELECT T3.organisation_type from Research_Staff as T1 join Organisations as T2 on T1.employer_organisation_id = T2.organisation_id join Organisation_Types as T3 on T2.organisation_type = T3.organisation_type WHERE T1.staff_details = UNKNOWN_VALUE	formula_1
SELECT T1.Food , Food from goods as T1 join receipts as T2 ORDER BY COUNT ( T2.* )	formula_1
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.semester LIMIT 1	video_game
SELECT SUM ( * ) from broadcast_share	formula_1
SELECT T2.Game_ID , T1.Market_district from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID	video_game
SELECT SUM ( T2.* ) from program as T1 join broadcast_share as T2 WHERE T1.Owner = UNKNOWN_VALUE	formula_1
SELECT T1.number , T4.country from drivers as T1 join lapTimes as T2 on T1.driverId = T2.driverId join races as T3 on T2.raceId = T3.raceId join circuits as T4 on T3.circuitId = T4.circuitId join results as T5 on T1.driverId = T5.driverId WHERE T5.fastestLapTime = UNKNOWN_VALUE	formula_1
SELECT SUM ( * ) from people	formula_1
SELECT SUM ( T1.circuitRef ) , T4.* from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId join lapTimes as T4 WHERE T3.fastestLapTime > UNKNOWN_VALUE	formula_1
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE ORDER BY LIMIT 1	formula_1
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.aid = UNKNOWN_VALUE	formula_1
SELECT SUM ( * ) from MovieTheaters	formula_1
SELECT series_name , series_name from TV_Channel	video_game
SELECT SUM ( * ) from Paragraphs	formula_1
SELECT T1.Version_Number from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID GROUP BY T1.Date_Effective_From HAVING T2.Document_Description	cre_Doc_Template_Mgt
SELECT dept_name from department ORDER BY LIMIT 1	cre_Doc_Template_Mgt
SELECT T2.* from Templates as T1 join Paragraphs as T2 GROUP BY T1.Date_Effective_From ORDER BY COUNT ( * ) LIMIT 1	cre_Doc_Template_Mgt
SELECT Title , Title from Songs WHERE Title = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.* from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.* , T1.Jul , * , Jul from temperature as T1 join hosting_city as T2	music_2
SELECT City from city WHERE City = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T1.SongId ORDER BY COUNT ( T3.* ) LIMIT 1	cre_Doc_Template_Mgt
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID WHERE T3.Claim_Type_Code = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T3.*	cre_Doc_Template_Mgt
SELECT SUM ( flno ) from flight	cre_Doc_Template_Mgt
SELECT Name from store	cre_Doc_Template_Mgt
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T4.organisation_type from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Organisation_Types as T4 on T3.organisation_type = T4.organisation_type ORDER BY T1.other_details	cre_Doc_Template_Mgt
SELECT SUM ( T2.alt ) , T3.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.url ORDER BY COUNT ( * ) LIMIT 1	cre_Doc_Template_Mgt
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	college_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT AVG ( T2.Production_code ) from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel WHERE T1.Hight_definition_TV = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.Origin from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T3.Rating_in_percent = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapSpeed > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.Sex , T2.Name , T2.Date_of_Birth from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID GROUP BY T1.People_ID	cre_Doc_Template_Mgt
SELECT Country_Id from country	cre_Doc_Template_Mgt
SELECT T1.Country from country as T1 join driver as T2 ORDER BY T2.Age	cre_Doc_Template_Mgt
SELECT AVG ( name ) from aircraft	cre_Doc_Template_Mgt
SELECT eg Agree Objectives from Tasks	cre_Doc_Template_Mgt
SELECT T2.* , T1.semester from section as T1 join prereq as T2 ORDER BY LIMIT 1	college_2
SELECT MIN ( Store_ID ) , MIN ( Store_ID ) from store WHERE Date_Opened > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.AlbumId from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T1.SongId ORDER BY COUNT ( T3.* ) LIMIT 1	cre_Doc_Template_Mgt
SELECT origin , destination from flight WHERE distance = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT id from languages ORDER BY name	cre_Doc_Template_Mgt
SELECT Store_ID from store WHERE Store_ID > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.Template_ID , T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T2.Document_Description ORDER BY COUNT ( T3.* ) LIMIT 1	cre_Doc_Template_Mgt
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Flavor like UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T2.Bandmate ORDER BY COUNT ( T3.* ) LIMIT 1	cre_Doc_Template_Mgt
SELECT Id from customers WHERE Id = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT Channel_ID from channel ORDER BY Channel_ID	cre_Doc_Template_Mgt
SELECT T1.product_id from Products as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT Version_Number from Templates	cre_Doc_Template_Mgt
SELECT Code , Code from Movies WHERE Code = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T3.Channel_ID from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T1.Name = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT AVG ( * ) from MovieTheaters	cre_Doc_Template_Mgt
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name < UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT aid from flight	cre_Doc_Template_Mgt
SELECT T2.Nov from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.City in UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.LastName like UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.Template_ID , T2.Document_Name from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID WHERE Template_ID = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.Version_Number from Templates as T1 join Paragraphs as T2 GROUP BY T1.Date_Effective_To HAVING T2.*	cre_Doc_Template_Mgt
SELECT AVG ( T1.aid ) , SUM ( T2.* ) from flight as T1 join certificate as T2 GROUP BY T1.origin	college_2
SELECT T2.name , T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id join time_slot as T6 GROUP BY T5.year HAVING T6.day	college_2
SELECT T1.destination from flight as T1 join certificate as T2 GROUP BY T1.aid HAVING T2.*	cre_Doc_Template_Mgt
SELECT alt , alt from circuits WHERE circuitRef > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT SUM ( T2.* ) from Templates as T1 join Paragraphs as T2 WHERE T1.Version_Number = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT SUM ( T3.* ) , T2.circuitRef from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.url	college_2
SELECT T4.Jul from match as T1 join hosting_city as T2 on T1.Match_ID = T2.Match_ID join city as T3 on T2.Host_City = T3.City_ID join temperature as T4 on T3.City_ID = T4.City_ID ORDER BY T1.Result LIMIT 1	cre_Doc_Template_Mgt
SELECT destination from flight	cre_Doc_Template_Mgt
SELECT MIN ( T2.customer_id ) from Addresses as T1 join Customers as T2 WHERE T1.address_id = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.product_price , T1.product_type_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id ORDER BY T3.order_date LIMIT 1	cre_Doc_Template_Mgt
SELECT T3.Press_ID from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE T1.Age like UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT T1.title from course as T1 join section as T2 on T1.course_id = T2.course_id WHERE T2.semester = UNKNOWN_VALUE	cre_Doc_Template_Mgt
SELECT MIN ( Quantity ) from stock	device
SELECT gender_code , customer_last_name from Customers WHERE country in UNKNOWN_VALUE	device
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	device
SELECT MIN ( order_id ) from Customer_Orders ORDER BY order_date	device
SELECT Id , Id from customers WHERE FirstName = UNKNOWN_VALUE	device
SELECT T4.alt from drivers as T1 join lapTimes as T2 on T1.driverId = T2.driverId join races as T3 on T2.raceId = T3.raceId join circuits as T4 on T3.circuitId = T4.circuitId ORDER BY T1.driverId	device
SELECT SUM ( T2.* ) from Customers as T1 join Shipment_Items as T2 WHERE T1.country in UNKNOWN_VALUE	device
SELECT T2.Jul from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.Hanzi = UNKNOWN_VALUE	device
SELECT T2.* from temperature as T1 join hosting_city as T2 ORDER BY T1.Aug LIMIT 1	device
SELECT Origin from program WHERE Name in UNKNOWN_VALUE	device
SELECT Official_native_language from country ORDER BY Capital	device
SELECT T1.Food from goods as T1 join items as T2 on T1.Id = T2.Item WHERE T2.Item > UNKNOWN_VALUE	device
SELECT organisation_id from Grants	device
SELECT Id from customers WHERE Id > UNKNOWN_VALUE	device
SELECT T2.title , T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join instructor as T3 on T1.dept_name = T3.dept_name join student as T4 on T1.dept_name = T4.dept_name join takes as T5 on T4.ID = T5.ID join section as T6 on T5.course_id = T6.course_id GROUP BY T6.year ORDER BY T3.ID LIMIT 1	device
SELECT T1.Country from country as T1 join driver as T2 WHERE T2.Points > UNKNOWN_VALUE	device
SELECT SUM ( * ) from Order_Items	device
SELECT T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY T2.ID LIMIT 1	device
SELECT T2.alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.raceId HAVING T3.*	device
SELECT destination , destination from flight WHERE aid < UNKNOWN_VALUE	device
SELECT customer_phone from Customers ORDER BY customer_email	device
SELECT T3.Press_ID , T1.Age from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID ORDER BY COUNT ( T1.Author_ID ) LIMIT 1	device
SELECT Press_ID from press ORDER BY Press_ID	device
SELECT T2.Directed_by from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel WHERE T1.Hight_definition_TV = UNKNOWN_VALUE	device
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID ORDER BY T2.Weight	device
SELECT T1.Headphone_ID , Headphone_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY T3.Neighborhood	e_commerce
SELECT SUM ( * ) from Paragraphs	device
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Price = UNKNOWN_VALUE	device
SELECT SUM ( * ) from Vocals	device
SELECT Title from Songs WHERE Title = UNKNOWN_VALUE	device
SELECT SUM ( T7.* ) , T6.outcome_details from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Projects as T5 on T4.organisation_id = T5.organisation_id join Project_Outcomes as T6 on T5.project_id = T6.project_id join Tasks as T7 GROUP BY T3.organisation_id ORDER BY T1.document_description	device
SELECT T1.Id from customers as T1 join goods as T2 join items as T3 join receipts as T4 WHERE T2.Price > UNKNOWN_VALUE GROUP BY T3.Ordinal HAVING T4.*	device
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	device
SELECT SUM ( T2.* ) from team as T1 join team_driver as T2 WHERE T1.Team_ID > UNKNOWN_VALUE	device
SELECT T3.Shop_ID , T3.Shop_Name from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE	device
SELECT Id from customers WHERE FirstName = UNKNOWN_VALUE	device
SELECT T1.Version_Number from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID GROUP BY T1.Template_ID HAVING T2.Document_Description	device
SELECT Neighborhood from store	device
SELECT MIN ( T1.Food ) , MIN ( Food ) , MIN ( Food ) from goods as T1 join receipts as T2 ORDER BY COUNT ( T2.* )	device
SELECT T4.organisation_type , T4.organisation_type_description from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Organisation_Types as T4 on T3.organisation_type = T4.organisation_type join Projects as T5 on T3.organisation_id = T5.organisation_id join Project_Outcomes as T6 on T5.project_id = T6.project_id GROUP BY T6.project_id ORDER BY T1.other_details LIMIT 1	device
SELECT T1.product_price from Products as T1 join Addresses as T2 WHERE T2.address_id = UNKNOWN_VALUE	device
SELECT destination from flight	device
SELECT MIN ( T2.country ) , MIN ( country ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join results as T3 on T1.raceId = T3.raceId GROUP BY T1.raceId ORDER BY T3.grid LIMIT 1	device
SELECT T1.name from instructor as T1 join advisor as T2 on T1.ID = T2.i_ID join student as T3 on T2.s_ID = T3.ID join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester = UNKNOWN_VALUE	device
SELECT Platform_name from platform	device
SELECT destination from flight GROUP BY distance HAVING distance	device
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	device
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	device
SELECT T1.Origin from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T3.Share_in_percent = UNKNOWN_VALUE	device
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join time_slot as T5 GROUP BY T4.year HAVING T5.start_hr	e_commerce
SELECT * from hosting_city	device
SELECT AVG ( dept_name ) , AVG ( dept_name ) from department	e_commerce
SELECT MIN ( order_id ) from Customer_Orders	device
SELECT AVG ( * ) from Claims_Processing	device
SELECT COUNT ( T2.* ) from Documents as T1 join Paragraphs as T2 WHERE T1.Document_ID = UNKNOWN_VALUE	device
SELECT name , id from languages	e_commerce
SELECT AVG ( T4.project_details ) from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Projects as T4 on T3.organisation_id = T4.organisation_id GROUP BY T1.other_details	device
SELECT AVG ( dept_name ) , AVG ( dept_name ) from department	e_commerce
SELECT T1.Name from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID ORDER BY T3.Press_ID	device
SELECT SUM ( T1.Food ) from goods as T1 join receipts as T2 GROUP BY Food ORDER BY COUNT ( T2.* ) LIMIT 1	device
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	e_commerce
SELECT T1.dept_name , dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name WHERE T2.course_id > UNKNOWN_VALUE	device
SELECT T3.Press_ID , T1.Age from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID GROUP BY T1.Gender	e_commerce
SELECT MIN ( * ) from people	device
SELECT T1.organisation_type , T4.outcome_details from Organisation_Types as T1 join Organisations as T2 on T1.organisation_type = T2.organisation_type join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Outcomes as T4 on T3.project_id = T4.project_id	cre_Doc_Template_Mgt
SELECT T1.Version_Number from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T2.Document_Name ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.id , id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id GROUP BY T3.politics_score	cre_Doc_Template_Mgt
SELECT T2.title , T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name LIMIT 1	cre_Doc_Template_Mgt
SELECT T5.* , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID join game_player as T5 WHERE T4.Player_ID = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.gender_code , T1.customer_last_name from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Id from customers as T1 join items as T2 WHERE T2.Ordinal > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId WHERE T2.StagePosition = UNKNOWN_VALUE	tracking_grants_for_research
SELECT destination from flight GROUP BY aid	tracking_grants_for_research
SELECT Oct , Oct from temperature ORDER BY Aug LIMIT 1	tracking_grants_for_research
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID join stock as T4 WHERE T1.Device in UNKNOWN_VALUE GROUP BY T3.Open_Date ORDER BY COUNT ( T4.* ) LIMIT 1	tracking_grants_for_research
SELECT dept_name , dept_name from department LIMIT 1	cre_Doc_Template_Mgt
SELECT T1.product_price from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id WHERE T3.order_status_code > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claims_Processing as T3 GROUP BY T2.Policy_ID ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT SUM ( T2.* ) from Customers as T1 join Order_Items as T2 GROUP BY T1.customer_number	tracking_grants_for_research
SELECT T1.gender_code , T1.customer_last_name , customer_last_name from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.name , T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name	cre_Doc_Template_Mgt
SELECT organisation_type from Organisation_Types	tracking_grants_for_research
SELECT SUM ( T2.alt ) , T4.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join results as T3 on T1.raceId = T3.raceId join lapTimes as T4 WHERE T3.fastestLapSpeed > UNKNOWN_VALUE GROUP BY T1.raceId HAVING *	tracking_grants_for_research
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.aid , T1.destination from flight as T1 join certificate as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.Id from customers as T1 join items as T2 WHERE T2.Ordinal > UNKNOWN_VALUE	tracking_grants_for_research
SELECT destination , destination from flight	cre_Doc_Template_Mgt
SELECT T1.Platform_name , T4.Player_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID WHERE T2.Developers = UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( T2.* ) from Customers as T1 join Shipment_Items as T2 WHERE T1.country in UNKNOWN_VALUE	tracking_grants_for_research
SELECT Store_ID from store	tracking_grants_for_research
SELECT T1.Store_ID from store as T1 join stock as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.Title from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie WHERE T2.Name = UNKNOWN_VALUE	tracking_grants_for_research
SELECT MIN ( T3.Car_# ) , COUNT ( T1.Make ) from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID	cre_Doc_Template_Mgt
SELECT SUM ( * ) from official_languages	tracking_grants_for_research
SELECT SUM ( T2.* ) , T1.email_address from Customers as T1 join Shipment_Items as T2 WHERE * = UNKNOWN_VALUE GROUP BY T1.country	tracking_grants_for_research
SELECT T2.title , T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join instructor as T3 on T1.dept_name = T3.dept_name ORDER BY T3.ID	tracking_grants_for_research
SELECT T2.order_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id join Order_Items as T3 GROUP BY T1.customer_number HAVING T3.*	tracking_grants_for_research
SELECT T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name WHERE T2.course_id > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claims_Processing as T3 GROUP BY T2.Policy_ID HAVING Customer_Details ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT T4.organisation_type , T1.other_details from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Organisation_Types as T4 on T3.organisation_type = T4.organisation_type join Tasks as T5 GROUP BY T3.organisation_details ORDER BY COUNT ( T5.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id WHERE T3.education_score = UNKNOWN_VALUE ORDER BY T3.overall_score LIMIT 1	tracking_grants_for_research
SELECT COUNT ( T2.* ) , T1.Platform_ID , Platform_ID from platform as T1 join game_player as T2	country_language
SELECT Shop_ID from shop ORDER BY LIMIT 1	tracking_grants_for_research
SELECT T1.dept_name , dept_name from department as T1 join time_slot as T2 ORDER BY T2.start_min LIMIT 1	tracking_grants_for_research
SELECT T1.organisation_type , T1.organisation_type_description from Organisation_Types as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	tracking_grants_for_research
SELECT organisation_type from Organisation_Types	tracking_grants_for_research
SELECT T1.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name ORDER BY T2.ID LIMIT 1	tracking_grants_for_research
SELECT product_id from Products	tracking_grants_for_research
SELECT T1.Country , Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Driver_ID ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT AVG ( T3.* ) , T1.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join prereq as T3 GROUP BY T2.ID HAVING ORDER BY LIMIT 1	country_language
SELECT T4.gender_code , gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* > UNKNOWN_VALUE GROUP BY T4.customer_id HAVING T1.product_id	tracking_grants_for_research
SELECT T1.name from instructor as T1 join advisor as T2 on T1.ID = T2.i_ID join student as T3 on T2.s_ID = T3.ID join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id join prereq as T6 GROUP BY T5.year ORDER BY COUNT ( T6.* ) LIMIT 1	tracking_grants_for_research
SELECT Version_Number from Templates WHERE Date_Effective_To = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.organisation_type from Organisation_Types as T1 join Organisations as T2 on T1.organisation_type = T2.organisation_type join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Staff as T4 on T3.project_id = T4.project_id WHERE T4.staff_id = UNKNOWN_VALUE	tracking_grants_for_research
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.Book_Series , T1.Press_ID from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID WHERE Book_Series = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.organisation_id from Grants as T1 join Organisations as T2 on T1.organisation_id = T2.organisation_id join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Staff as T4 on T3.project_id = T4.project_id ORDER BY T4.date_from	tracking_grants_for_research
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Food > UNKNOWN_VALUE	tracking_grants_for_research
SELECT AVG ( dept_name ) from department ORDER BY LIMIT 1	tracking_grants_for_research
SELECT T1.Title , T3.AId from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T2.AlbumId = UNKNOWN_VALUE ORDER BY T3.Year LIMIT 1	tracking_grants_for_research
SELECT AVG ( T1.dept_name ) from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.name , T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.salary from flight as T1 join employee as T2 join certificate as T3 GROUP BY T1.arrival_date ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT Customer_Details from Customers	tracking_grants_for_research
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.aid = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.organisation_type from Grants as T1 join Organisations as T2 on T1.organisation_id = T2.organisation_id join Organisation_Types as T3 on T2.organisation_type = T3.organisation_type WHERE T1.grant_end_date > UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( T3.driverRef ) , T4.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join lapTimes as T4 GROUP BY T1.raceId HAVING *	formula_1
SELECT T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.course_id > UNKNOWN_VALUE GROUP BY T5.year HAVING course_id	tracking_grants_for_research
SELECT T3.Store_ID , T3.Name , Name from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID ORDER BY T1.Earpads LIMIT 1	tracking_grants_for_research
SELECT document_id from Documents	tracking_grants_for_research
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name < UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( * ) from MovieTheaters	tracking_grants_for_research
SELECT Document_Description , Document_Name , Document_Name from Documents	college_2
SELECT Food from goods WHERE Price > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.Title from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID WHERE T1.Name > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.Version_Number , T1.Template_Type_Code , Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code WHERE T2.Date_Effective_From = UNKNOWN_VALUE	tracking_grants_for_research
SELECT Hight_definition_TV from TV_Channel WHERE id = UNKNOWN_VALUE	tracking_grants_for_research
SELECT MIN ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Description = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.* , T1.Author_ID from author as T1 join book as T2 GROUP BY T1.Gender	formula_1
SELECT T2.* from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.Position from Performance as T1 join Songs as T2 on T1.SongId = T2.SongId join Tracklists as T3 on T2.SongId = T3.SongId WHERE T1.StagePosition = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T3.Shop_Name > UNKNOWN_VALUE GROUP BY T1.Software_Platform HAVING T1.Device	tracking_grants_for_research
SELECT Channel_ID from channel ORDER BY Rating_in_percent LIMIT 1	tracking_grants_for_research
SELECT T1.dept_name from department as T1 join time_slot as T2 WHERE T2.day > UNKNOWN_VALUE	tracking_grants_for_research
SELECT * from MovieTheaters	tracking_grants_for_research
SELECT T2.name , name from flight as T1 join aircraft as T2 WHERE T1.aid = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Id from customers as T1 join goods as T2 ORDER BY T2.Price	tracking_grants_for_research
SELECT T2.* from match as T1 join hosting_city as T2 ORDER BY T1.Result LIMIT 1	tracking_grants_for_research
SELECT T2.Other_Details from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 on T2.Document_ID = T3.Document_ID GROUP BY T1.Template_ID HAVING T3.Paragraph_Text	tracking_grants_for_research
SELECT SUM ( * ) from Order_Items	tracking_grants_for_research
SELECT T1.Hight_definition_TV from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel WHERE T2.Weekly_Rank = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.aid = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T5.* , T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year	college_2
SELECT T2.Title from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID WHERE T1.Name > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.year , T3.dob , dob from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join circuits as T4 on T1.circuitId = T4.circuitId join lapTimes as T5 WHERE T4.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T5.* LIMIT 1	tracking_grants_for_research
SELECT gender_code , customer_last_name , customer_last_name from Customers	college_2
SELECT SUM ( * ) from Paragraphs	tracking_grants_for_research
SELECT AVG ( T2.Production_code ) from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel GROUP BY T1.Language	tracking_grants_for_research
SELECT AVG ( T2.* ) from TV_Channel as T1 join Cartoon as T2 WHERE T1.Hight_definition_TV = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.alt , alt from qualifying as T1 join races as T2 on T1.raceId = T2.raceId join circuits as T3 on T2.circuitId = T3.circuitId ORDER BY T1.qualifyId	tracking_grants_for_research
SELECT T1.Country from country as T1 join team as T2 WHERE T2.Team_ID = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T4.Jul , T5.* , T4.Oct from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID join hosting_city as T5 WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Score LIMIT 1	tracking_grants_for_research
SELECT T1.Code from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie join MovieTheaters as T3 GROUP BY T2.Name ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT Title from game	tracking_grants_for_research
SELECT T3.Type from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId join Vocals as T4 GROUP BY T1.SongId ORDER BY COUNT ( T4.* ) LIMIT 1	tracking_grants_for_research
SELECT SUM ( T2.customer_phone ) , T3.* from Addresses as T1 join Customers as T2 join Order_Items as T3 GROUP BY T1.address_id	formula_1
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY T3.Name HAVING T1.Driver-matched_dB	tracking_grants_for_research
SELECT gender_code , gender_code , customer_last_name from Customers	college_2
SELECT alt , country from circuits WHERE circuitRef = UNKNOWN_VALUE	tracking_grants_for_research
SELECT Game_ID , Title from game	formula_1
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	formula_1
SELECT AVG ( StagePosition ) from Performance	tracking_grants_for_research
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE GROUP BY T4.year ORDER BY T4.time_slot_id LIMIT 1	tracking_grants_for_research
SELECT Title from Songs WHERE Title like UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( flno ) from flight	tracking_grants_for_research
SELECT dept_name from department	tracking_grants_for_research
SELECT SUM ( T5.dept_name ) , T1.room_number from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name	formula_1
SELECT id from languages ORDER BY name	tracking_grants_for_research
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Flavor > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id WHERE T3.health_score like UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Customer_ID from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID join Claims_Documents as T4 on T3.Claim_Header_ID = T4.Claim_ID ORDER BY T4.Claim_ID LIMIT 1	tracking_grants_for_research
SELECT T1.Press_ID , T1.Month_Profits_billion from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID ORDER BY T2.Sale_Amount LIMIT 1	tracking_grants_for_research
SELECT SUM ( T2.* ) , T1.outcome_details from Project_Outcomes as T1 join Tasks as T2 ORDER BY COUNT ( * )	tracking_grants_for_research
SELECT Id from customers	tracking_grants_for_research
SELECT T1.year , T3.dob from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId WHERE T1.raceId > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Food > UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( T3.* ) , T2.circuitRef from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId	tracking_grants_for_research
SELECT T3.project_details from Staff_Roles as T1 join Project_Staff as T2 on T1.role_code = T2.role_code join Projects as T3 on T2.project_id = T3.project_id WHERE T1.role_description > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T2.Bandmate ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT Country from country	car_racing
SELECT SUM ( T3.driverRef ) , T5.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join circuits as T4 on T1.circuitId = T4.circuitId join lapTimes as T5 WHERE T4.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId	car_racing
SELECT T1.Game_ID from game as T1 join game_player as T2 on T1.Game_ID = T2.Game_ID join player as T3 on T2.Player_ID = T3.Player_ID GROUP BY T3.Player_ID HAVING T3.College	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT T3.Type from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT T1.alt , alt from circuits as T1 join lapTimes as T2 join races as T3 on T1.circuitId = T3.circuitId join results as T4 on T3.raceId = T4.raceId WHERE T4.fastestLapSpeed > UNKNOWN_VALUE ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T6.title , T5.dept_name from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name join course as T6 on T2.course_id = T6.course_id GROUP BY T6.course_id ORDER BY SUM ( T1.room_number ) LIMIT 1	car_racing
SELECT SUM ( T2.* ) from Addresses as T1 join Order_Items as T2 WHERE T1.address_id = UNKNOWN_VALUE	car_racing
SELECT T6.date_to from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Projects as T5 on T4.organisation_id = T5.organisation_id join Project_Staff as T6 on T5.project_id = T6.project_id WHERE T1.document_description = UNKNOWN_VALUE	car_racing
SELECT Unsure_rate from candidate ORDER BY Consider_rate LIMIT 1	car_racing
SELECT SUM ( T1.aid ) , T2.* from flight as T1 join certificate as T2 GROUP BY aid	formula_1
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	car_racing
SELECT SUM ( T3.* ) , T1.Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Driver	formula_1
SELECT SUM ( T2.* ) , T1.Applications from device as T1 join stock as T2 GROUP BY T1.Device_ID	formula_1
SELECT T1.gender_code , gender_code , T1.customer_last_name from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	car_racing
SELECT T5.organisation_type from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type join Projects as T6 on T4.organisation_id = T6.organisation_id join Project_Staff as T7 on T6.project_id = T7.project_id WHERE T1.document_description = UNKNOWN_VALUE ORDER BY T7.date_from	car_racing
SELECT T2.Franchise , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID	formula_1
SELECT SUM ( * ) from team_driver	car_racing
SELECT MIN ( Movie ) from MovieTheaters	car_racing
SELECT AVG ( StagePosition ) from Performance	car_racing
SELECT AVG ( T1.aid ) from flight as T1 join employee as T2 WHERE T2.salary < UNKNOWN_VALUE	car_racing
SELECT T2.Franchise , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID WHERE T4.Player_name = UNKNOWN_VALUE ORDER BY Player_name	car_racing
SELECT SUM ( T3.* ) , T1.Year_Profits_billion from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID join book as T3 GROUP BY T2.Book_Series	formula_1
SELECT T1.customer_email , T2.order_id , T1.customer_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id	video_game
SELECT SUM ( T2.Document_Description ) , T3.* from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T1.Template_Details	formula_1
SELECT T2.Book_Series from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID ORDER BY T1.Press_ID	car_racing
SELECT SUM ( * ) from MovieTheaters	car_racing
SELECT T1.organisation_type from Organisation_Types as T1 join Organisations as T2 on T1.organisation_type = T2.organisation_type join Projects as T3 on T2.organisation_id = T3.organisation_id join Project_Staff as T4 on T3.project_id = T4.project_id ORDER BY T4.date_from	car_racing
SELECT T2.eid from flight as T1 join employee as T2 join certificate as T3 GROUP BY T1.price ORDER BY COUNT ( T3.* ) LIMIT 1	car_racing
SELECT Unsure_rate from candidate WHERE People_ID in UNKNOWN_VALUE	car_racing
SELECT AVG ( T1.Production_code ) from Cartoon as T1 join Cartoon as T2 GROUP BY T1.Original_air_date ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	car_racing
SELECT T1.Parking from store as T1 join stock as T2 GROUP BY T1.Date_Opened ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT T1.origin from flight as T1 join certificate as T2 GROUP BY T1.distance ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Model in UNKNOWN_VALUE	car_racing
SELECT AVG ( T2.* ) , T1.dept_name , dept_name from department as T1 join prereq as T2	video_game
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE ORDER BY LIMIT 1	car_racing
SELECT product_type_code from Products	car_racing
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE	car_racing
SELECT SUM ( T1.Original_air_date ) , T2.* from Cartoon as T1 join Cartoon as T2 GROUP BY T1.Production_code	formula_1
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT SUM ( * ) from MovieTheaters	car_racing
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 WHERE T1.Title = UNKNOWN_VALUE ORDER BY COUNT ( T3.* ) LIMIT 1	car_racing
SELECT T1.dept_name , dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id join prereq as T6 WHERE T2.course_id > UNKNOWN_VALUE GROUP BY T5.year HAVING T6.* ORDER BY LIMIT 1	car_racing
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	car_racing
SELECT project_id from Project_Outcomes	car_racing
SELECT MIN ( T4.customer_id ) , MIN ( customer_id ) from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id ORDER BY T1.product_type_code	car_racing
SELECT Version_Number from Templates WHERE Date_Effective_From = UNKNOWN_VALUE	car_racing
SELECT T3.organisation_type from Research_Staff as T1 join Organisations as T2 on T1.employer_organisation_id = T2.organisation_id join Organisation_Types as T3 on T2.organisation_type = T3.organisation_type join Tasks as T4 GROUP BY T1.employer_organisation_id ORDER BY COUNT ( T4.* ) LIMIT 1	car_racing
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.Id > UNKNOWN_VALUE	car_racing
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT T3.Press_ID from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE T1.Name = UNKNOWN_VALUE	car_racing
SELECT T1.year , T3.country , country from races as T1 join qualifying as T2 on T1.raceId = T2.raceId join circuits as T3 on T1.circuitId = T3.circuitId GROUP BY T1.raceId HAVING T2.q3	video_game
SELECT Program_ID from program	car_racing
SELECT T2.* from Movies as T1 join MovieTheaters as T2 ORDER BY T1.Code	car_racing
SELECT SUM ( * ) from Tasks	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT SUM ( T3.* ) , T1.Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Driver_ID	formula_1
SELECT T1.Title from Songs as T1 join Vocals as T2 GROUP BY T1.SongId ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT SUM ( T3.* ) , T1.Staff_ID , * from Staff as T1 join Claims_Processing as T2 on T1.Staff_ID = T2.Staff_ID join Claims_Processing as T3 GROUP BY T2.Claim_Outcome_Code	video_game
SELECT Unsure_rate , Oppose_rate from candidate GROUP BY Unsure_rate ORDER BY Oppose_rate	car_racing
SELECT AVG ( T1.Hight_definition_TV ) , MIN ( Hight_definition_TV ) from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel GROUP BY T2.Channel	formula_1
SELECT organisation_type from Organisation_Types	car_racing
SELECT T1.Country from country as T1 join team as T2 WHERE T2.Team_ID = UNKNOWN_VALUE	car_racing
SELECT AVG ( s_ID ) from advisor	car_racing
SELECT T2.AlbumId from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	car_racing
SELECT T2.name , T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year HAVING dept_name	college_2
SELECT T1.gender_code , T1.customer_last_name , customer_last_name from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	car_racing
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT T5.* , T4.Jul from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID join hosting_city as T5 WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Result LIMIT 1	car_racing
SELECT T1.Title from Songs as T1 join Vocals as T2 WHERE Title = UNKNOWN_VALUE ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T5.* , T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year	music_2
SELECT T1.alt , T1.country from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV like UNKNOWN_VALUE	car_racing
SELECT SUM ( T3.* ) , T2.Name from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie join MovieTheaters as T3 GROUP BY T1.Title	college_2
SELECT organisation_type from Organisation_Types	car_racing
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id ORDER BY T3.education_score LIMIT 1	car_racing
SELECT AVG ( T1.dept_name ) , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year LIMIT 1	college_2
SELECT AVG ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT T5.organisation_type from Staff_Roles as T1 join Project_Staff as T2 on T1.role_code = T2.role_code join Projects as T3 on T2.project_id = T3.project_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type join Tasks as T6 GROUP BY T1.role_description ORDER BY COUNT ( T6.* ) LIMIT 1	car_racing
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT T1.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name ORDER BY T2.ID LIMIT 1	car_racing
SELECT SUM ( * ) from team_driver	car_racing
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.LastName like UNKNOWN_VALUE	car_racing
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE GROUP BY T4.year HAVING year	car_racing
SELECT T1.id from languages as T1 join official_languages as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	car_racing
SELECT T5.* , T4.Jul from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID join hosting_city as T5 WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Result LIMIT 1	car_racing
SELECT SUM ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	car_racing
SELECT T2.AlbumId from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	car_racing
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year ORDER BY COUNT ( T5.* ) LIMIT 1	car_racing
SELECT T4.* , T1.country from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId join lapTimes as T4 WHERE T3.fastestLapTime > UNKNOWN_VALUE	car_racing
SELECT SUM ( T2.* ) from Addresses as T1 join Order_Items as T2 WHERE T1.address_details = UNKNOWN_VALUE	car_racing
SELECT Hight_definition_TV , Hight_definition_TV from TV_Channel	college_2
SELECT SUM ( T2.* ) , T1.product_type_code from Products as T1 join Order_Items as T2 GROUP BY product_type_code	college_2
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Price > UNKNOWN_VALUE	car_racing
SELECT SUM ( Id ) from customers	car_racing
SELECT Store_ID from store WHERE Parking > UNKNOWN_VALUE	car_racing
SELECT SUM ( T2.* ) , T1.Launch from program as T1 join broadcast_share as T2 GROUP BY T1.Owner	college_2
SELECT SUM ( T2.* ) from Addresses as T1 join Order_Items as T2 WHERE T1.address_id = UNKNOWN_VALUE	car_racing
SELECT T1.Version_Number from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID WHERE T2.Document_ID = UNKNOWN_VALUE	car_racing
SELECT T1.aid , T1.destination from flight as T1 join aircraft as T2 WHERE T2.name < UNKNOWN_VALUE	car_racing
SELECT Code from Movies	car_racing
SELECT T1.Id , Id from customers as T1 join receipts as T2 WHERE T1.LastName = UNKNOWN_VALUE ORDER BY COUNT ( T2.* )	car_racing
SELECT T1.Origin from program as T1 join broadcast_share as T2 GROUP BY T1.Owner ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T2.AlbumId from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	car_racing
SELECT T1.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY T2.ID LIMIT 1	car_racing
SELECT T3.Press_ID from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE T1.Name like UNKNOWN_VALUE	car_racing
SELECT T1.overall_score from countries as T1 join official_languages as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name ORDER BY T2.course_id LIMIT 1	car_racing
SELECT MIN ( T2.Name ) from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID WHERE T1.Unsure_rate < UNKNOWN_VALUE	car_racing
SELECT AVG ( dept_name ) from department	car_racing
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name = UNKNOWN_VALUE	car_racing
SELECT SUM ( * ) from Order_Items	car_racing
SELECT T2.* from team as T1 join team_driver as T2 WHERE T1.Team_ID = UNKNOWN_VALUE	car_racing
SELECT MIN ( T2.alt ) , MIN ( T2.lat ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T3.*	car_racing
SELECT id from languages	car_racing
SELECT customer_phone from Customers	car_racing
SELECT T1.Origin from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID WHERE T3.Channel_ID = UNKNOWN_VALUE	car_racing
SELECT SUM ( Headphone_ID ) , Headphone_ID from headphone	college_2
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year ORDER BY SUM ( T5.* ) LIMIT 1	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT MIN ( T1.url ) , SUM ( T4.* ) from races as T1 join qualifying as T2 on T1.raceId = T2.raceId join circuits as T3 on T1.circuitId = T3.circuitId join lapTimes as T4 WHERE T3.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING * ORDER BY SUM ( T2.qualifyId ) LIMIT 1	car_racing
SELECT MIN ( T2.country ) , MIN ( country ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 join results as T4 on T1.raceId = T4.raceId WHERE T4.fastestLapSpeed > UNKNOWN_VALUE GROUP BY T1.raceId HAVING T3.*	car_racing
SELECT T1.Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Driver HAVING T3.*	car_racing
SELECT dept_name , dept_name from department	college_2
SELECT T2.* , T1.Jul from temperature as T1 join hosting_city as T2 GROUP BY T1.Feb ORDER BY COUNT ( * ) LIMIT 1	car_racing
SELECT organisation_type from Organisation_Types	tracking_grants_for_research
SELECT T1.Id , Id , T2.Price from customers as T1 join goods as T2 join receipts as T3 ORDER BY COUNT ( T3.* )	tracking_grants_for_research
SELECT dept_name , dept_name from department LIMIT 1	bakery_1
SELECT T2.Nov from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.City = UNKNOWN_VALUE	tracking_grants_for_research
SELECT * from people	tracking_grants_for_research
SELECT SUM ( * ) from official_languages	tracking_grants_for_research
SELECT T4.position , T3.dob from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join constructorStandings as T4 on T1.raceId = T4.raceId join results as T5 on T1.raceId = T5.raceId WHERE T5.fastestLapSpeed > UNKNOWN_VALUE GROUP BY T1.raceId	tracking_grants_for_research
SELECT T1.Id from customers as T1 join receipts as T2 on T1.Id = T2.CustomerId WHERE T2.ReceiptNumber = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID join Claims_Processing as T4 GROUP BY T3.Claim_Header_ID ORDER BY COUNT ( T4.* ) LIMIT 1	tracking_grants_for_research
SELECT AVG ( dept_name ) from department ORDER BY LIMIT 1	tracking_grants_for_research
SELECT Version_Number from Templates	tracking_grants_for_research
SELECT Platform_ID from platform WHERE Platform_ID = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T4.gender_code , T4.customer_last_name from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T2.name from time_slot as T1 join instructor as T2 ORDER BY SUM ( T1.end_hr )	tracking_grants_for_research
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T2.StagePosition ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.series_name , T1.Language from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel ORDER BY T2.Episode	tracking_grants_for_research
SELECT T1.People_ID from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID ORDER BY T2.Sex LIMIT 1	tracking_grants_for_research
SELECT T2.* , T1.Hanzi from city as T1 join hosting_city as T2 WHERE Hanzi > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Template_Type_Code , Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID WHERE T3.Other_Details = UNKNOWN_VALUE	tracking_grants_for_research
SELECT * from MovieTheaters	tracking_grants_for_research
SELECT T1.Customer_Details from Customers as T1 join Claims_Processing as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	tracking_grants_for_research
SELECT MIN ( T1.Food ) , MIN ( Food ) from goods as T1 join receipts as T2 GROUP BY T2.ReceiptNumber	bakery_1
SELECT T1.gender_code , T3.order_item_id , order_item_id from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Order_Items as T3 on T2.order_id = T3.order_id	college_2
SELECT SUM ( T2.* ) from results as T1 join lapTimes as T2 WHERE T1.fastestLapTime > UNKNOWN_VALUE	tracking_grants_for_research
SELECT dept_name , dept_name from department	bakery_1
SELECT customer_phone from Customers	tracking_grants_for_research
SELECT T1.Country from country as T1 join driver as T2 join team_driver as T3 GROUP BY T2.Car_# ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT Document_Description from Documents	tracking_grants_for_research
SELECT parent_product_id , parent_product_id from Products WHERE product_description = UNKNOWN_VALUE	tracking_grants_for_research
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( flno ) from flight	tracking_grants_for_research
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 GROUP BY T3.economics_score ORDER BY COUNT ( T4.* ) LIMIT 1	tracking_grants_for_research
SELECT Origin from program WHERE Name like UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.* , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 WHERE T2.Units_sold_Millions = UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( * ) from Order_Items	tracking_grants_for_research
SELECT T3.dept_name from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join department as T3 on T2.dept_name = T3.dept_name join teaches as T4 on T2.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY T1.i_ID LIMIT 1	tracking_grants_for_research
SELECT T2.Book_Series from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID WHERE T1.Name > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.City from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID join hosting_city as T3 GROUP BY T2.Mar ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT Origin from program	tracking_grants_for_research
SELECT destination , destination from flight ORDER BY price LIMIT 1	tracking_grants_for_research
SELECT T4.gender_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 on T2.order_item_id = T5.order_item_id join Shipments as T6 on T5.shipment_id = T6.shipment_id join Invoices as T7 on T6.invoice_number = T7.invoice_number WHERE T1.product_description > UNKNOWN_VALUE GROUP BY T7.invoice_status_code HAVING T1.product_id	tracking_grants_for_research
SELECT Country from country	tracking_grants_for_research
SELECT Id from customers WHERE Id = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID join stock as T4 GROUP BY T1.Headphone_ID ORDER BY COUNT ( T4.* ) LIMIT 1	tracking_grants_for_research
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID ORDER BY T1.Software_Platform	tracking_grants_for_research
SELECT T2.eid , T2.salary from flight as T1 join employee as T2 ORDER BY T1.destination	tracking_grants_for_research
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name = UNKNOWN_VALUE	tracking_grants_for_research
SELECT City from city WHERE City = UNKNOWN_VALUE	tracking_grants_for_research
SELECT destination , destination from flight	bakery_1
SELECT Shop_ID from shop WHERE Open_Year > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T1.Code from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie join MovieTheaters as T3 GROUP BY T2.Movie ORDER BY COUNT ( T3.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.Id from customers as T1 join receipts as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	tracking_grants_for_research
SELECT T1.gender_code , gender_code from Customers as T1 join Shipment_Items as T2 GROUP BY T1.customer_id ORDER BY COUNT ( T2.* )	tracking_grants_for_research
SELECT SUM ( T2.alt ) , T4.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join results as T3 on T1.raceId = T3.raceId join lapTimes as T4 GROUP BY T1.url ORDER BY T3.grid LIMIT 1	tracking_grants_for_research
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Class > UNKNOWN_VALUE	tracking_grants_for_research
SELECT T4.county , T1.parent_product_id , T4.customer_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	tracking_grants_for_research
SELECT T5.dept_name , dept_name from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name ORDER BY T1.room_number LIMIT 1	tracking_grants_for_research
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( Headphone_ID ) from headphone	tracking_grants_for_research
SELECT SUM ( T1.alt ) , T4.* from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId join lapTimes as T4 WHERE T3.fastestLapTime > UNKNOWN_VALUE GROUP BY T3.milliseconds ORDER BY COUNT ( * )	tracking_grants_for_research
SELECT T1.county from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	tracking_grants_for_research
SELECT SUM ( Headphone_ID ) from headphone	tracking_grants_for_research
SELECT id from languages	tracking_grants_for_research
SELECT Title from Movies WHERE Code = UNKNOWN_VALUE	movie_2
SELECT id from languages ORDER BY name	movie_2
SELECT * from MovieTheaters	movie_2
SELECT parent_product_id , product_name from Products WHERE product_description = UNKNOWN_VALUE	movie_2
SELECT MIN ( T3.Car_# ) , COUNT ( T1.Make ) from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID	country_language
SELECT * from Shipment_Items	movie_2
SELECT SUM ( T2.Other_Details ) , T3.* from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T1.Date_Effective_From	country_language
SELECT T1.politics_score from countries as T1 join official_languages as T2 on T1.id = T2.country_id GROUP BY T2.country_id HAVING T1.name	movie_2
SELECT organisation_type from Organisation_Types	movie_2
SELECT Origin from program	movie_2
SELECT T3.Press_ID , T1.Age from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID	country_language
SELECT organisation_id from Grants	movie_2
SELECT T3.Position from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId join Tracklists as T3 on T1.SongId = T3.SongId join Vocals as T4 WHERE T1.Title = UNKNOWN_VALUE GROUP BY T2.StagePosition ORDER BY COUNT ( T4.* ) LIMIT 1	movie_2
SELECT T1.Name , T3.Year_Profits_billion from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE Name > UNKNOWN_VALUE GROUP BY Year_Profits_billion HAVING T1.Author_ID	movie_2
SELECT destination , destination from flight	country_language
SELECT T1.alt , alt from circuits as T1 join lapTimes as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	movie_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	movie_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 WHERE T1.Title = UNKNOWN_VALUE ORDER BY COUNT ( T3.* ) LIMIT 1	movie_2
SELECT Shop_ID from shop WHERE Shop_ID = UNKNOWN_VALUE	movie_2
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID ORDER BY T1.Model	movie_2
SELECT T1.dept_name , dept_name from department as T1 join prereq as T2 join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID GROUP BY T4.grade ORDER BY COUNT ( T2.* ) LIMIT 1	movie_2
SELECT T1.Id , Id from customers as T1 join goods as T2 WHERE T2.Flavor = UNKNOWN_VALUE	movie_2
SELECT T1.organisation_type from Organisation_Types as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	movie_2
SELECT SUM ( * ) from Paragraphs	movie_2
SELECT MIN ( T2.* ) , T1.Height , T1.Date_of_Birth from people as T1 join people as T2	movie_2
SELECT SUM ( T1.product_id ) , T2.* from Products as T1 join Shipment_Items as T2	country_language
SELECT T2.Jul from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.City = UNKNOWN_VALUE	movie_2
SELECT Instrument from Instruments	movie_2
SELECT product_type_code from Products ORDER BY product_type_code	movie_2
SELECT T2.Year from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City WHERE T1.City = UNKNOWN_VALUE	movie_2
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID ORDER BY T2.Name LIMIT 1	movie_2
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	movie_2
SELECT T2.Version_Number , T1.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code	country_language
SELECT MIN ( T2.lat ) , MIN ( lat ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId HAVING circuitRef	movie_2
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code WHERE T1.Template_Type_Code like UNKNOWN_VALUE	movie_2
SELECT T3.title from classroom as T1 join section as T2 on T1.building = T2.building join course as T3 on T2.course_id = T3.course_id ORDER BY T1.room_number LIMIT 1	movie_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	movie_2
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE ORDER BY LIMIT 1	movie_2
SELECT T1.gender_code , gender_code from Customers as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	movie_2
SELECT T1.product_type_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id WHERE T3.order_status_code > UNKNOWN_VALUE	movie_2
SELECT dept_name from department	movie_2
SELECT SUM ( * ) from team_driver	movie_2
SELECT T3.Shop_ID , T3.Shop_Name from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE	movie_2
SELECT T2.project_id from Tasks as T1 join Project_Outcomes as T2 join Project_Staff as T3 GROUP BY T3.date_from ORDER BY COUNT ( T1.* ) LIMIT 1	movie_2
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	movie_2
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	movie_2
SELECT T1.title from course as T1 join section as T2 on T1.course_id = T2.course_id join prereq as T3 GROUP BY T2.year ORDER BY COUNT ( T3.* ) LIMIT 1	movie_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	movie_2
SELECT destination , destination from flight WHERE distance < UNKNOWN_VALUE	movie_2
SELECT SUM ( T2.alt ) , T3.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId ORDER BY COUNT ( * )	movie_2
SELECT AVG ( T1.customer_phone ) , T2.order_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id WHERE order_id = UNKNOWN_VALUE	movie_2
SELECT T2.* from team as T1 join team_driver as T2 WHERE T1.Team_ID = UNKNOWN_VALUE	movie_2
SELECT AVG ( T3.dept_name ) , AVG ( T1.s_ID ) from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join department as T3 on T2.dept_name = T3.dept_name join teaches as T4 on T2.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year	country_language
SELECT T2.AlbumId from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	movie_2
SELECT SUM ( T2.position ) , T3.* from races as T1 join constructorStandings as T2 on T1.raceId = T2.raceId join lapTimes as T3 GROUP BY T1.raceId	country_language
SELECT document_type_code from Document_Types	movie_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	movie_2
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.SongId = UNKNOWN_VALUE	movie_2
SELECT AVG ( flno ) , aid from flight	country_language
SELECT T1.People_ID , T2.Name from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID LIMIT 1	country_language
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID join people as T3 GROUP BY T2.Sex ORDER BY COUNT ( T3.* ) LIMIT 1	movie_2
SELECT T2.Franchise from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID WHERE T1.Platform_ID in UNKNOWN_VALUE	movie_2
SELECT T2.Title from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID WHERE T1.Platform_ID in UNKNOWN_VALUE	movie_2
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID GROUP BY T3.Date_of_Settlement ORDER BY LIMIT 1	movie_2
SELECT T1.name from instructor as T1 join advisor as T2 on T1.ID = T2.i_ID join student as T3 on T2.s_ID = T3.ID join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE	college_2
SELECT Type from Albums	college_2
SELECT AVG ( T1.dept_name ) , AVG ( dept_name ) from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	college_2
SELECT T3.Store_ID , T1.Headphone_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID GROUP BY Store_ID	music_2
SELECT AVG ( T1.dept_name ) , AVG ( dept_name ) , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year	college_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	college_2
SELECT T4.Jul from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Venue	college_2
SELECT sec_id from section GROUP BY year ORDER BY LIMIT 1	college_2
SELECT Id from customers	college_2
SELECT T3.* , T2.country from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T1.raceId = UNKNOWN_VALUE	college_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	college_2
SELECT customer_address from Customers	college_2
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	college_2
SELECT Customer_Details from Customers	college_2
SELECT T4.position , T3.dob from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join constructorStandings as T4 on T1.raceId = T4.raceId join results as T5 on T1.raceId = T5.raceId WHERE T5.fastestLapSpeed > UNKNOWN_VALUE GROUP BY T1.raceId	college_2
SELECT T1.Country , T2.Team_ID from country as T1 join team as T2 join team_driver as T3 GROUP BY T1.Country_Id ORDER BY COUNT ( T3.* ) LIMIT 1	college_2
SELECT AVG ( Document_Name ) , COUNT ( Document_Name ) from Documents	music_2
SELECT product_type_code from Products	college_2
SELECT Food from goods WHERE Price = UNKNOWN_VALUE	college_2
SELECT AVG ( T1.Template_Type_Code ) , SUM ( T3.* ) from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Version_Number	music_2
SELECT T2.Food , T1.Id from customers as T1 join goods as T2 WHERE T2.Flavor > UNKNOWN_VALUE	college_2
SELECT T3.Winnings from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID WHERE T1.Team_ID = UNKNOWN_VALUE	college_2
SELECT Title from game WHERE Units_sold_Millions = UNKNOWN_VALUE	college_2
SELECT T1.Name from author as T1 join book as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	college_2
SELECT name from instructor	college_2
SELECT T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name WHERE T2.course_id > UNKNOWN_VALUE	college_2
SELECT T3.Shop_ID , Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID ORDER BY T1.Device_ID	college_2
SELECT organisation_id from Grants	college_2
SELECT SUM ( * ) from Cartoon	college_2
SELECT SUM ( T3.* ) , T1.Country from country as T1 join team as T2 join team_driver as T3 GROUP BY T2.Team	music_2
SELECT T2.Franchise , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID ORDER BY T1.Market_district	college_2
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	college_2
SELECT AVG ( T2.name ) from flight as T1 join aircraft as T2 WHERE T1.aid < UNKNOWN_VALUE	college_2
SELECT destination , destination from flight	music_2
SELECT AVG ( T1.product_description ) from Products as T1 join Shipment_Items as T2 WHERE T2.* = UNKNOWN_VALUE	college_2
SELECT T2.Position from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId WHERE T1.Label = UNKNOWN_VALUE	college_2
SELECT eid from employee	college_2
SELECT * from Paragraphs	college_2
SELECT T1.Type from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId join Songs as T3 on T2.SongId = T3.SongId join Instruments as T4 on T3.SongId = T4.SongId WHERE T4.Instrument = UNKNOWN_VALUE	college_2
SELECT destination from flight	college_2
SELECT Id , Id from customers WHERE Id in UNKNOWN_VALUE	college_2
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id	music_2
SELECT T1.Food , Food , Food from goods as T1 join items as T2 on T1.Id = T2.Item ORDER BY T2.Receipt	college_2
SELECT T1.product_type_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id WHERE T3.order_date > UNKNOWN_VALUE	college_2
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID WHERE T3.Claim_Type_Code like UNKNOWN_VALUE	college_2
SELECT Channel_ID from channel ORDER BY Rating_in_percent LIMIT 1	college_2
SELECT SUM ( T5.organisation_type_description ) , T6.* from Research_Outcomes as T1 join Project_Outcomes as T2 on T1.outcome_code = T2.outcome_code join Projects as T3 on T2.project_id = T3.project_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type join Tasks as T6 GROUP BY T1.outcome_description ORDER BY COUNT ( * ) LIMIT 1	college_2
SELECT MIN ( T1.Date_of_Birth ) , MIN ( T1.Name ) from people as T1 join people as T2 GROUP BY T1.Height ORDER BY COUNT ( T2.* ) LIMIT 1	college_2
SELECT T2.Title from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID ORDER BY T1.Market_district	college_2
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id WHERE T3.education_score > UNKNOWN_VALUE	college_2
SELECT AVG ( T1.product_description ) , SUM ( T4.* ) from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Shipment_Items as T4 GROUP BY T3.order_id	music_2
SELECT AVG ( aid ) from flight	college_2
SELECT MIN ( Device_ID ) from device	college_2
SELECT SUM ( T2.* ) , T1.Name from people as T1 join people as T2 WHERE T1.Weight < UNKNOWN_VALUE	college_2
SELECT SUM ( Headphone_ID ) from headphone	college_2
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	college_2
SELECT T2.Food from customers as T1 join goods as T2 WHERE T1.FirstName = UNKNOWN_VALUE	college_2
SELECT AVG ( T2.* ) from Orders as T1 join Shipment_Items as T2 GROUP BY T1.date_order_placed	college_2
SELECT product_price , product_type_code from Products	music_2
SELECT MIN ( T2.order_id ) , COUNT ( order_id ) from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id GROUP BY T1.payment_method_code	music_2
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.SongId = UNKNOWN_VALUE	college_2
SELECT Code from Movies WHERE Title = UNKNOWN_VALUE	college_2
SELECT T1.Title , T3.AId from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId join Vocals as T4 WHERE T1.SongId = UNKNOWN_VALUE ORDER BY COUNT ( T4.* ) LIMIT 1	college_2
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 WHERE T1.Title = UNKNOWN_VALUE ORDER BY COUNT ( T3.* ) LIMIT 1	college_2
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	college_2
SELECT Position , Player_ID from player WHERE Player_name = UNKNOWN_VALUE	college_2
SELECT SUM ( * ) from Paragraphs	college_2
SELECT SUM ( * ) from Order_Items	college_2
SELECT Template_Type_Code from Ref_Template_Types WHERE Template_Type_Description = UNKNOWN_VALUE	college_2
SELECT AVG ( T1.Hight_definition_TV ) from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel GROUP BY T2.Channel	college_2
SELECT SUM ( T2.* ) from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	college_2
SELECT Template_Type_Code from Ref_Template_Types WHERE Template_Type_Description = UNKNOWN_VALUE	college_2
SELECT MIN ( T2.customer_id ) , COUNT ( customer_id ) from Addresses as T1 join Customers as T2 WHERE T1.address_id = UNKNOWN_VALUE	college_2
SELECT SUM ( * ) from Shipment_Items	college_2
SELECT SUM ( T2.Other_Details ) , T3.* from Templates as T1 join Documents as T2 on T1.Template_ID = T2.Template_ID join Paragraphs as T3 GROUP BY T1.Template_Details	video_game
SELECT MIN ( Name ) from store	college_2
SELECT T1.Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId GROUP BY T2.StagePosition	college_2
SELECT Id from customers WHERE FirstName = UNKNOWN_VALUE	college_2
SELECT Owner from program	college_2
SELECT SUM ( * ) from Order_Items	college_2
SELECT T1.Hight_definition_TV , T2.Channel from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel join Cartoon as T3 ORDER BY COUNT ( T3.* ) LIMIT 1	college_2
SELECT T2.Jul from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.Hanzi = UNKNOWN_VALUE	college_2
SELECT T1.Instrument from Instruments as T1 join Songs as T2 on T1.SongId = T2.SongId join Performance as T3 on T2.SongId = T3.SongId join Vocals as T4 GROUP BY T3.StagePosition ORDER BY COUNT ( T4.* ) LIMIT 1	college_2
SELECT Press_ID from press	college_2
SELECT T1.Id from customers as T1 join items as T2 WHERE T2.Ordinal > UNKNOWN_VALUE	college_2
SELECT T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name WHERE T2.course_id > UNKNOWN_VALUE	college_2
SELECT T5.* , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	college_2
SELECT T1.Title , Title from Songs as T1 join Performance as T2 on T1.SongId = T2.SongId WHERE Title = UNKNOWN_VALUE ORDER BY T2.Bandmate LIMIT 1	college_2
SELECT Hight_definition_TV from TV_Channel ORDER BY id	college_2
SELECT SUM ( T5.* ) , SUM ( * ) , T4.customer_name from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T1.product_type_code	cre_Doc_Template_Mgt
SELECT SUM ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Description = UNKNOWN_VALUE	college_2
SELECT SUM ( T2.* ) from races as T1 join lapTimes as T2 WHERE T1.raceId > UNKNOWN_VALUE	college_2
SELECT SUM ( * ) from Shipment_Items	college_2
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Flavor = UNKNOWN_VALUE	college_2
SELECT T2.AlbumId from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Vocals as T3 GROUP BY T1.SongId ORDER BY COUNT ( T3.* ) LIMIT 1	college_2
SELECT SUM ( T2.* ) from team as T1 join team_driver as T2 WHERE T1.Team_ID = UNKNOWN_VALUE	college_2
SELECT Title from Songs WHERE Title = UNKNOWN_VALUE	college_2
SELECT SUM ( T2.Claim_Stage_ID ) , T3.* from Claim_Headers as T1 join Claims_Processing_Stages as T2 join Claims_Processing as T3 GROUP BY T1.Amount_Claimed	video_game
SELECT T1.destination from flight as T1 join aircraft as T2 GROUP BY T2.name	college_2
SELECT SUM ( T2.* ) , T1.Country from country as T1 join team_driver as T2 GROUP BY T1.Regoin	video_game
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	college_2
SELECT distance from flight	college_2
SELECT MIN ( T3.Document_Name ) , COUNT ( Document_Name ) from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID WHERE T1.Template_Type_Description = UNKNOWN_VALUE	college_2
SELECT T1.i_ID from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join teaches as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY T2.ID LIMIT 1	college_2
SELECT MIN ( T1.alt ) from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	college_2
SELECT T1.Press_ID from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID join book as T3 GROUP BY T2.Release_date ORDER BY COUNT ( T3.* ) LIMIT 1	college_2
SELECT T3.Type from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T1.Title = UNKNOWN_VALUE	college_2
SELECT T3.* , T2.order_id from Addresses as T1 join Customer_Orders as T2 join Order_Items as T3 WHERE T1.address_id = UNKNOWN_VALUE	college_2
SELECT T1.product_type_code , product_type_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T4.customer_id = UNKNOWN_VALUE	college_2
SELECT T1.Hight_definition_TV from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel WHERE T2.Viewers_m = UNKNOWN_VALUE	college_2
SELECT SUM ( T4.* ) , T3.Match_ID from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join hosting_city as T4 WHERE T1.City in UNKNOWN_VALUE	college_2
SELECT People_ID from candidate WHERE People_ID in UNKNOWN_VALUE	college_2
SELECT AVG ( T2.* ) from TV_Channel as T1 join Cartoon as T2 WHERE T1.Hight_definition_TV = UNKNOWN_VALUE	college_2
SELECT T4.Paragraph_Text from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID join Paragraphs as T4 on T3.Document_ID = T4.Document_ID WHERE T1.Template_Type_Code = UNKNOWN_VALUE	college_2
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE	college_2
SELECT SUM ( * ) from people	college_2
SELECT destination from flight	college_2
SELECT SUM ( * ) from Shipment_Items	college_2
SELECT SUM ( T4.* ) , T1.Launch from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID join broadcast_share as T4 GROUP BY T3.Channel_ID	video_game
SELECT SUM ( T2.* ) , T1.order_id from Customer_Orders as T1 join Order_Items as T2	video_game
SELECT T1.Hight_definition_TV from TV_Channel as T1 join TV_series as T2 on T1.id = T2.Channel WHERE T2.Share = UNKNOWN_VALUE	college_2
SELECT SongId from Songs WHERE Title = UNKNOWN_VALUE	college_2
SELECT SUM ( T4.* ) , T3.outcome_details from Project_Staff as T1 join Projects as T2 on T1.project_id = T2.project_id join Project_Outcomes as T3 on T2.project_id = T3.project_id join Tasks as T4 GROUP BY T1.date_to	video_game
SELECT destination from flight	college_2
SELECT Id from customers WHERE FirstName = UNKNOWN_VALUE	college_2
SELECT Id , Id from customers	video_game
SELECT T4.* , T3.Car_# from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID join team_driver as T4 WHERE T1.Team_ID > UNKNOWN_VALUE	college_2
SELECT T1.Game_ID from game as T1 join game_player as T2 on T1.Game_ID = T2.Game_ID join player as T3 on T2.Player_ID = T3.Player_ID GROUP BY T3.Player_ID HAVING T1.Units_sold_Millions	college_2
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID WHERE T3.Claim_Type_Code like UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( T2.* ) from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	insurance_and_eClaims
SELECT Id from customers WHERE Id = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.destination from flight as T1 join certificate as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	insurance_and_eClaims
SELECT Unsure_rate from candidate ORDER BY People_ID LIMIT 1	insurance_and_eClaims
SELECT MIN ( T2.* ) , MIN ( T1.Height ) , Height , Height from people as T1 join people as T2 GROUP BY T1.Sex	flight_1
SELECT AlbumId from Tracklists	insurance_and_eClaims
SELECT T1.Code from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie WHERE T2.Name = UNKNOWN_VALUE	insurance_and_eClaims
SELECT * from MovieTheaters	insurance_and_eClaims
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( * ) from stock	insurance_and_eClaims
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	insurance_and_eClaims
SELECT series_name from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	insurance_and_eClaims
SELECT Unsure_rate from candidate ORDER BY Date LIMIT 1	insurance_and_eClaims
SELECT T1.Name , T3.Year_Profits_billion from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE Name > UNKNOWN_VALUE GROUP BY T1.Age HAVING T1.Author_ID	insurance_and_eClaims
SELECT Unsure_rate from candidate ORDER BY Date	insurance_and_eClaims
SELECT T2.organisation_id , T1.other_details from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Projects as T4 on T3.organisation_id = T4.organisation_id join Tasks as T5 on T4.project_id = T5.project_id WHERE T5.eg Agree Objectives > UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( T2.alt ) , alt , T3.* from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 WHERE T2.circuitRef > UNKNOWN_VALUE GROUP BY T1.raceId ORDER BY COUNT ( * )	insurance_and_eClaims
SELECT T2.customer_phone from Addresses as T1 join Customers as T2 join Order_Items as T3 GROUP BY T1.address_id ORDER BY COUNT ( T3.* ) LIMIT 1	insurance_and_eClaims
SELECT T4.customer_phone from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T1.product_type_code ORDER BY COUNT ( T5.* ) LIMIT 1	insurance_and_eClaims
SELECT Document_ID from Documents	insurance_and_eClaims
SELECT T2.Jul from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.Hanzi > UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.Origin , T3.Channel_ID , Channel_ID from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID	bakery_1
SELECT T2.eid from flight as T1 join employee as T2 join certificate as T3 GROUP BY T1.arrival_date ORDER BY COUNT ( T3.* ) LIMIT 1	insurance_and_eClaims
SELECT Unsure_rate from candidate WHERE People_ID = UNKNOWN_VALUE ORDER BY Date	insurance_and_eClaims
SELECT SUM ( T1.id ) , T2.* from languages as T1 join official_languages as T2	movie_2
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE ORDER BY LIMIT 1	insurance_and_eClaims
SELECT MIN ( Movie ) from MovieTheaters	insurance_and_eClaims
SELECT gender_code , customer_id from Customers	movie_2
SELECT AVG ( Hight_definition_TV ) from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T2.Position from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId WHERE T1.Title = UNKNOWN_VALUE	insurance_and_eClaims
SELECT Claim_Type_Code from Claim_Headers GROUP BY Claim_Header_ID	insurance_and_eClaims
SELECT SUM ( * ) from team_driver	insurance_and_eClaims
SELECT T3.Store_ID , Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Price = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.eg Agree Objectives from Tasks as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	insurance_and_eClaims
SELECT T1.origin from flight as T1 join certificate as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	insurance_and_eClaims
SELECT T1.destination from flight as T1 join employee as T2 GROUP BY T1.aid HAVING T2.salary	insurance_and_eClaims
SELECT Shop_ID from shop ORDER BY Location	insurance_and_eClaims
SELECT T1.product_price from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T4.customer_id > UNKNOWN_VALUE	insurance_and_eClaims
SELECT T3.* , * , T2.order_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id join Order_Items as T3 GROUP BY T1.customer_number	bakery_1
SELECT LastName from customers	insurance_and_eClaims
SELECT T4.gender_code , T1.product_description from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 WHERE T5.* = UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( * ) from Vocals	insurance_and_eClaims
SELECT T4.Player_name from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 on T2.Game_ID = T3.Game_ID join player as T4 on T3.Player_ID = T4.Player_ID WHERE T1.Platform_ID = UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( T3.driverRef ) , T4.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join lapTimes as T4 GROUP BY T1.raceId	movie_2
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	insurance_and_eClaims
SELECT destination , destination from flight WHERE destination = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.alt , alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	insurance_and_eClaims
SELECT T2.name , name from flight as T1 join aircraft as T2 ORDER BY T1.price LIMIT 1	insurance_and_eClaims
SELECT T1.Title from game as T1 join game_player as T2 on T1.Game_ID = T2.Game_ID join player as T3 on T2.Player_ID = T3.Player_ID ORDER BY T3.Player_ID	insurance_and_eClaims
SELECT T1.dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY T2.ID LIMIT 1	insurance_and_eClaims
SELECT Code from Movies WHERE Title = UNKNOWN_VALUE	insurance_and_eClaims
SELECT SUM ( Flavor ) , Food from goods ORDER BY Food LIMIT 1	insurance_and_eClaims
SELECT T1.Food from goods as T1 join receipts as T2 GROUP BY Food ORDER BY COUNT ( T2.* ) LIMIT 1	insurance_and_eClaims
SELECT T3.Claim_Type_Code from Staff as T1 join Claims_Processing as T2 on T1.Staff_ID = T2.Staff_ID join Claim_Headers as T3 on T2.Claim_ID = T3.Claim_Header_ID WHERE T1.Staff_Details = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.name from instructor as T1 join advisor as T2 on T1.ID = T2.i_ID join student as T3 on T2.s_ID = T3.ID join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id WHERE T5.semester > UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T3.grade ORDER BY T4.year LIMIT 1	insurance_and_eClaims
SELECT dept_name from department ORDER BY LIMIT 1	insurance_and_eClaims
SELECT AVG ( T1.Next_Claim_Stage_ID ) from Claims_Processing_Stages as T1 join Claims_Processing as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	insurance_and_eClaims
SELECT SUM ( * ) from Vocals	insurance_and_eClaims
SELECT T2.* from candidate as T1 join people as T2 GROUP BY T1.Unsure_rate ORDER BY COUNT ( * ) LIMIT 1	insurance_and_eClaims
SELECT T3.Shop_ID , T3.Shop_Name from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID ORDER BY T1.Device_ID	insurance_and_eClaims
SELECT T4.gender_code , gender_code from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description = UNKNOWN_VALUE	insurance_and_eClaims
SELECT Shop_ID from shop WHERE Open_Date = UNKNOWN_VALUE	insurance_and_eClaims
SELECT T1.Country from country as T1 join driver as T2 ORDER BY T2.Age LIMIT 1	car_racing
SELECT MIN ( invoice_number ) from Invoices	car_racing
SELECT T2.alt , alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.raceId ORDER BY COUNT ( T3.* )	car_racing
SELECT T1.Store_ID from store as T1 join stock as T2 GROUP BY Store_ID ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T2.* , T1.Team_ID from team as T1 join team_driver as T2 GROUP BY T1.Make	e_commerce
SELECT series_name , Language from TV_Channel ORDER BY id	car_racing
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	car_racing
SELECT AVG ( T1.flno ) , flno , T2.name from flight as T1 join aircraft as T2	formula_1
SELECT Official_native_language from country ORDER BY Capital	car_racing
SELECT T2.salary , T2.eid , T1.aid from flight as T1 join employee as T2 ORDER BY T1.distance	car_racing
SELECT SUM ( * ) from MovieTheaters	car_racing
SELECT Unsure_rate from candidate WHERE Unsure_rate < UNKNOWN_VALUE	car_racing
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE ORDER BY LIMIT 1	car_racing
SELECT T1.url , T3.dob from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join results as T4 on T1.raceId = T4.raceId WHERE T4.fastestLapTime > UNKNOWN_VALUE	car_racing
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 WHERE T3.justice_score = UNKNOWN_VALUE GROUP BY T3.politics_score ORDER BY COUNT ( T4.* ) LIMIT 1	car_racing
SELECT origin , flno from flight	e_commerce
SELECT product_id from Products	car_racing
SELECT T1.Code from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie join MovieTheaters as T3 GROUP BY T2.Name HAVING T3.*	car_racing
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Price > UNKNOWN_VALUE	car_racing
SELECT Hight_definition_TV , Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT MIN ( Car_# ) , COUNT ( Car_# ) from driver	e_commerce
SELECT T1.Country from country as T1 join driver as T2 ORDER BY T2.Age	car_racing
SELECT MIN ( T3.order_id ) from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Order_Items as T4 GROUP BY T1.product_type_code ORDER BY COUNT ( T4.* ) LIMIT 1	car_racing
SELECT SUM ( * ) from game_player	car_racing
SELECT name from aircraft	car_racing
SELECT COUNT ( T2.* ) , T1.Car_# , Car_# from driver as T1 join team_driver as T2	formula_1
SELECT dept_name from department ORDER BY LIMIT 1	car_racing
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join time_slot as T5 GROUP BY T4.year ORDER BY SUM ( T5.start_hr ) LIMIT 1	car_racing
SELECT Code from Movies WHERE Title = UNKNOWN_VALUE	car_racing
SELECT T3.Match_ID from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID WHERE T1.City = UNKNOWN_VALUE	car_racing
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.aid = UNKNOWN_VALUE	car_racing
SELECT T1.Name from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID ORDER BY T2.Title LIMIT 1	car_racing
SELECT AVG ( aid ) from flight	car_racing
SELECT T1.Origin , T3.Channel_ID from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID	e_commerce
SELECT SUM ( T2.* ) , T1.Id from customers as T1 join receipts as T2	e_commerce
SELECT Position from Tracklists	car_racing
SELECT series_name from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT Franchise from game	car_racing
SELECT SUM ( * ) from Claims_Processing	car_racing
SELECT T1.Country from country as T1 join team as T2 WHERE T2.Team_ID > UNKNOWN_VALUE	car_racing
SELECT Name , Age from author GROUP BY Name ORDER BY COUNT ( Author_ID ) LIMIT 1	car_racing
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	e_commerce
SELECT City from city WHERE Hanzi > UNKNOWN_VALUE	car_racing
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	car_racing
SELECT T1.Unsure_rate from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID WHERE T2.Name < UNKNOWN_VALUE	car_racing
SELECT AVG ( Hight_definition_TV ) from TV_Channel GROUP BY Language	car_racing
SELECT MIN ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Description = UNKNOWN_VALUE	car_racing
SELECT SUM ( * ) from Tasks	car_racing
SELECT T1.AlbumId from Tracklists as T1 join Vocals as T2 GROUP BY T1.Position ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT AVG ( T2.project_details ) from Project_Outcomes as T1 join Projects as T2 on T1.project_id = T2.project_id GROUP BY T1.project_id	car_racing
SELECT T1.product_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Customer_Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Order_Items as T5 GROUP BY T4.customer_name ORDER BY COUNT ( T5.* ) LIMIT 1	car_racing
SELECT T1.AlbumId from Tracklists as T1 join Vocals as T2 GROUP BY T1.Position ORDER BY COUNT ( T2.* ) LIMIT 1	car_racing
SELECT T1.destination from flight as T1 join aircraft as T2 WHERE T2.name > UNKNOWN_VALUE	car_racing
SELECT T4.* , T1.Team_ID from team as T1 join team_driver as T2 on T1.Team_ID = T2.Team_ID join driver as T3 on T2.Driver_ID = T3.Driver_ID join team_driver as T4 GROUP BY T3.Driver_ID	e_commerce
SELECT T1.customer_email , T2.order_id , T1.customer_id from Customers as T1 join Customer_Orders as T2 on T1.customer_id = T2.customer_id	formula_1
SELECT T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id	e_commerce
SELECT SUM ( * ) from official_languages	car_racing
SELECT T5.organisation_type from Staff_Roles as T1 join Project_Staff as T2 on T1.role_code = T2.role_code join Projects as T3 on T2.project_id = T3.project_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Organisation_Types as T5 on T4.organisation_type = T5.organisation_type join Project_Outcomes as T6 on T3.project_id = T6.project_id WHERE T1.role_description > UNKNOWN_VALUE GROUP BY T3.project_id HAVING project_id	car_racing
SELECT SUM ( * ) from game_player	car_racing
SELECT MIN ( id ) , MIN ( id ) from TV_Channel	e_commerce
SELECT SUM ( T3.driverRef ) , T4.* from races as T1 join lapTimes as T2 on T1.raceId = T2.raceId join drivers as T3 on T2.driverId = T3.driverId join lapTimes as T4 GROUP BY T1.raceId	e_commerce
SELECT Customer_Details from Customers	car_racing
SELECT Store_ID from store ORDER BY Name	car_racing
SELECT T1.product_price , T5.customer_id from Products as T1 join Addresses as T2 join Order_Items as T3 on T1.product_id = T3.product_id join Customer_Orders as T4 on T3.order_id = T4.order_id join Customers as T5 on T4.customer_id = T5.customer_id GROUP BY T2.address_id HAVING T3.order_item_id	e_commerce
SELECT T1.Id , Id from customers as T1 join receipts as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT Staff_Details from Staff WHERE Staff_Details = UNKNOWN_VALUE	bakery_1
SELECT SUM ( T1.alt ) , T2.* from circuits as T1 join lapTimes as T2 WHERE T1.circuitRef > UNKNOWN_VALUE	bakery_1
SELECT People_ID from candidate WHERE Unsure_rate < UNKNOWN_VALUE	bakery_1
SELECT AlbumId from Tracklists	bakery_1
SELECT T4.gender_code , T4.customer_last_name from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id WHERE T1.product_description = UNKNOWN_VALUE	bakery_1
SELECT T3.Year , T4.* , T1.Jul from temperature as T1 join city as T2 on T1.City_ID = T2.City_ID join hosting_city as T3 on T2.City_ID = T3.Host_City join hosting_city as T4 WHERE T1.Nov like UNKNOWN_VALUE ORDER BY T1.Aug LIMIT 1	bakery_1
SELECT * from MovieTheaters	bakery_1
SELECT T1.Origin from program as T1 join broadcast_share as T2 GROUP BY T1.Name ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT Title from Movies	bakery_1
SELECT AVG ( T3.dept_name ) , AVG ( T1.s_ID ) from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join department as T3 on T2.dept_name = T3.dept_name join teaches as T4 on T2.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY LIMIT 1	insurance_and_eClaims
SELECT date_to from Project_Staff	bakery_1
SELECT AVG ( Owner ) from program	bakery_1
SELECT COUNT ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	bakery_1
SELECT T1.dept_name , dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.year ORDER BY LIMIT 1	formula_1
SELECT MIN ( T1.Channel ) , MIN ( Channel ) from TV_series as T1 join Cartoon as T2 GROUP BY T1.Weekly_Rank ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT T3.* , T2.sec_id from course as T1 join section as T2 on T1.course_id = T2.course_id join prereq as T3 WHERE T1.course_id > UNKNOWN_VALUE	bakery_1
SELECT AVG ( T1.dept_name ) , AVG ( dept_name ) from department as T1 join course as T2 on T1.dept_name = T2.dept_name GROUP BY T2.course_id	insurance_and_eClaims
SELECT Country , Country from country	insurance_and_eClaims
SELECT dept_name from department	bakery_1
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	bakery_1
SELECT Document_Description , Document_Name , Document_Name from Documents	formula_1
SELECT T5.* , T4.sec_id , T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year	formula_1
SELECT T1.Staff_Details from Staff as T1 join Claims_Processing as T2 on T1.Staff_ID = T2.Staff_ID join Claim_Headers as T3 on T2.Claim_ID = T3.Claim_Header_ID join Policies as T4 on T3.Policy_ID = T4.Policy_ID GROUP BY T4.Start_Date HAVING T3.Date_of_Settlement	bakery_1
SELECT SUM ( T5.dept_name ) , T1.room_number from classroom as T1 join section as T2 on T1.building = T2.building join takes as T3 on T2.course_id = T3.course_id join student as T4 on T3.ID = T4.ID join department as T5 on T4.dept_name = T5.dept_name GROUP BY T2.year	insurance_and_eClaims
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Class > UNKNOWN_VALUE	bakery_1
SELECT T1.Customer_Details from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID ORDER BY T3.Claim_Header_ID	bakery_1
SELECT T2.Name from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie ORDER BY T1.Rating	bakery_1
SELECT Origin from program ORDER BY Program_ID	bakery_1
SELECT Title from Songs WHERE Title = UNKNOWN_VALUE	bakery_1
SELECT T5.organisation_id from Research_Outcomes as T1 join Project_Outcomes as T2 on T1.outcome_code = T2.outcome_code join Projects as T3 on T2.project_id = T3.project_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Grants as T5 on T4.organisation_id = T5.organisation_id join Tasks as T6 GROUP BY T1.outcome_description ORDER BY COUNT ( T6.* ) LIMIT 1	bakery_1
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	bakery_1
SELECT MIN ( Sale_Amount ) , MIN ( Sale_Amount ) , MIN ( Sale_Amount ) from book GROUP BY Sale_Amount	formula_1
SELECT T3.* , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID join game_player as T3 WHERE T2.Developers = UNKNOWN_VALUE	bakery_1
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.fastestLapTime > UNKNOWN_VALUE	bakery_1
SELECT T2.title , T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year ORDER BY year	bakery_1
SELECT MIN ( order_id ) , COUNT ( order_id ) from Customer_Orders	insurance_and_eClaims
SELECT Id from customers WHERE FirstName in UNKNOWN_VALUE	bakery_1
SELECT AVG ( aid ) from flight	bakery_1
SELECT MIN ( customer_id ) from Customers	bakery_1
SELECT T1.title from course as T1 join section as T2 on T1.course_id = T2.course_id WHERE T2.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	bakery_1
SELECT T2.alt from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join lapTimes as T3 GROUP BY T1.raceId HAVING T3.*	bakery_1
SELECT T1.Food from goods as T1 join items as T2 on T1.Id = T2.Item ORDER BY T2.Ordinal LIMIT 1	bakery_1
SELECT SUM ( T2.* ) from Albums as T1 join Vocals as T2 WHERE T1.Label = UNKNOWN_VALUE	bakery_1
SELECT T1.Template_Type_Code , Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Template_Details ORDER BY COUNT ( T3.* ) LIMIT 1	bakery_1
SELECT Franchise from game WHERE Developers = UNKNOWN_VALUE	bakery_1
SELECT T1.title from course as T1 join section as T2 on T1.course_id = T2.course_id WHERE T2.semester > UNKNOWN_VALUE	bakery_1
SELECT T2.Version_Number from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code WHERE T1.Template_Type_Code = UNKNOWN_VALUE	bakery_1
SELECT origin from flight WHERE distance = UNKNOWN_VALUE	bakery_1
SELECT SUM ( * ) from Vocals	bakery_1
SELECT AVG ( T1.aid ) from flight as T1 join aircraft as T2 WHERE T2.name > UNKNOWN_VALUE	bakery_1
SELECT T1.Title , Title from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T3.Label = UNKNOWN_VALUE	bakery_1
SELECT origin , destination from flight WHERE distance = UNKNOWN_VALUE	bakery_1
SELECT SUM ( * ) from game_player	bakery_1
SELECT MIN ( Press_ID ) , MIN ( Press_ID ) from press	insurance_and_eClaims
SELECT MIN ( T1.Flavor ) from goods as T1 join receipts as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	bakery_1
SELECT T3.order_id from Invoices as T1 join Shipments as T2 on T1.invoice_number = T2.invoice_number join Orders as T3 on T2.order_id = T3.order_id join Shipment_Items as T4 WHERE T4.* = UNKNOWN_VALUE GROUP BY T1.invoice_date ORDER BY COUNT ( * ) LIMIT 1	bakery_1
SELECT Title , Title from Songs WHERE Title = UNKNOWN_VALUE	bakery_1
SELECT AVG ( aid ) from flight WHERE aid < UNKNOWN_VALUE	bakery_1
SELECT dept_name , dept_name from department LIMIT 1	insurance_and_eClaims
SELECT T3.Channel_ID from program as T1 join broadcast_share as T2 on T1.Program_ID = T2.Program_ID join channel as T3 on T2.Channel_ID = T3.Channel_ID ORDER BY T1.Origin	bakery_1
SELECT SUM ( * ) from official_languages	bakery_1
SELECT T3.Document_Description from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Documents as T3 on T2.Template_ID = T3.Template_ID GROUP BY T1.Template_Type_Description HAVING T3.Other_Details	bakery_1
SELECT T2.Claim_Type_Code from Policies as T1 join Claim_Headers as T2 on T1.Policy_ID = T2.Policy_ID join Claims_Processing as T3 GROUP BY T1.Policy_ID ORDER BY COUNT ( T3.* ) LIMIT 1	bakery_1
SELECT AVG ( T1.SongId ) from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T3.Label = UNKNOWN_VALUE	music_2
SELECT Press_ID from press	music_2
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID WHERE T1.Device = UNKNOWN_VALUE	music_2
SELECT * from MovieTheaters	music_2
SELECT SUM ( T2.* ) from Movies as T1 join MovieTheaters as T2 WHERE T1.Title = UNKNOWN_VALUE	music_2
SELECT Title from Songs	music_2
SELECT T3.Store_ID from headphone as T1 join stock as T2 on T1.Headphone_ID = T2.Headphone_ID join store as T3 on T2.Store_ID = T3.Store_ID WHERE T1.Model in UNKNOWN_VALUE	music_2
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Food > UNKNOWN_VALUE	music_2
SELECT Customer_Details from Customers	music_2
SELECT MIN ( order_quantity ) from Order_Items	music_2
SELECT T2.Flavor , T1.Id from customers as T1 join goods as T2 join items as T3 ORDER BY T3.Receipt LIMIT 1	music_2
SELECT SUM ( T2.* ) , T1.outcome_details from Project_Outcomes as T1 join Tasks as T2 GROUP BY T1.project_id	book_press
SELECT Shop_ID from shop WHERE Shop_ID = UNKNOWN_VALUE	music_2
SELECT Code from Movies WHERE Code = UNKNOWN_VALUE	music_2
SELECT T1.Id , Id from customers as T1 join goods as T2 join receipts as T3 GROUP BY T2.Price ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT SUM ( T2.* ) from Tracklists as T1 join Vocals as T2 WHERE T1.AlbumId = UNKNOWN_VALUE	music_2
SELECT T3.alt from qualifying as T1 join races as T2 on T1.raceId = T2.raceId join circuits as T3 on T2.circuitId = T3.circuitId ORDER BY T1.qualifyId	music_2
SELECT T2.salary from flight as T1 join employee as T2 WHERE T1.distance = UNKNOWN_VALUE	music_2
SELECT T2.organisation_id , T1.other_details from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Tasks as T3 join Organisations as T4 on T2.organisation_id = T4.organisation_id join Projects as T5 on T4.organisation_id = T5.organisation_id join Project_Staff as T6 on T5.project_id = T6.project_id GROUP BY T6.staff_id ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT AVG ( dept_name ) from department	music_2
SELECT destination , destination from flight	book_press
SELECT T4.Instrument from Albums as T1 join Tracklists as T2 on T1.AId = T2.AlbumId join Songs as T3 on T2.SongId = T3.SongId join Instruments as T4 on T3.SongId = T4.SongId WHERE T1.Label = UNKNOWN_VALUE	music_2
SELECT SUM ( T2.qualifyId ) , T3.* from races as T1 join qualifying as T2 on T1.raceId = T2.raceId join lapTimes as T3 GROUP BY T1.url ORDER BY SUM ( T1.time ) LIMIT 1	music_2
SELECT Country from country	music_2
SELECT T3.Match_ID from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID WHERE T1.City = UNKNOWN_VALUE	music_2
SELECT T1.organisation_type , T1.organisation_type_description , organisation_type_description from Organisation_Types as T1 join Tasks as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	music_2
SELECT SUM ( T2.* ) , T1.Platform_ID from platform as T1 join game_player as T2	book_press
SELECT T1.gender_code from Customers as T1 join Orders as T2 on T1.customer_id = T2.customer_id join Shipment_Items as T3 WHERE T3.* > UNKNOWN_VALUE GROUP BY T2.date_order_placed HAVING *	music_2
SELECT organisation_type from Organisation_Types	music_2
SELECT T1.alt from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join results as T3 on T2.raceId = T3.raceId WHERE T3.milliseconds > UNKNOWN_VALUE	music_2
SELECT T3.dept_name , dept_name , dept_name from advisor as T1 join instructor as T2 on T1.i_ID = T2.ID join department as T3 on T2.dept_name = T3.dept_name join teaches as T4 on T2.ID = T4.ID join section as T5 on T4.course_id = T5.course_id join course as T6 on T5.course_id = T6.course_id GROUP BY T6.course_id HAVING T1.i_ID ORDER BY LIMIT 1	device
SELECT T2.salary from flight as T1 join employee as T2 ORDER BY T1.aid LIMIT 1	music_2
SELECT SUM ( * ) from Paragraphs	music_2
SELECT T1.Customer_Details from Customers as T1 join Claims_Processing as T2 ORDER BY COUNT ( T2.* ) LIMIT 1	music_2
SELECT SUM ( T2.* ) , T1.Name from author as T1 join book as T2 WHERE Name > UNKNOWN_VALUE GROUP BY T1.Gender	music_2
SELECT T3.organisation_id from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id WHERE T1.document_type_code = UNKNOWN_VALUE	music_2
SELECT Id from customers WHERE FirstName = UNKNOWN_VALUE	music_2
SELECT T1.Food from goods as T1 join receipts as T2 join receipts as T3 GROUP BY T2.ReceiptNumber ORDER BY COUNT ( T3.* ) LIMIT 1	music_2
SELECT T2.Franchise from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID WHERE T1.Platform_ID = UNKNOWN_VALUE	music_2
SELECT AVG ( outcome_details ) from Project_Outcomes	music_2
SELECT T4.alt from drivers as T1 join lapTimes as T2 on T1.driverId = T2.driverId join races as T3 on T2.raceId = T3.raceId join circuits as T4 on T3.circuitId = T4.circuitId WHERE T1.driverId > UNKNOWN_VALUE	music_2
SELECT Id from customers	music_2
SELECT product_type_code from Products	music_2
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV like UNKNOWN_VALUE	music_2
SELECT Id , Id from customers	book_press
SELECT T2.Year from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City WHERE T1.Hanzi = UNKNOWN_VALUE	music_2
SELECT MIN ( other_details ) from Documents	music_2
SELECT Store_ID , Name from store	book_press
SELECT T1.alt from circuits as T1 join lapTimes as T2 ORDER BY COUNT ( T2.* )	music_2
SELECT Origin from program ORDER BY Origin	music_2
SELECT * from MovieTheaters	music_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id join prereq as T5 GROUP BY T4.year ORDER BY COUNT ( T5.* ) LIMIT 1	music_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester < UNKNOWN_VALUE	music_2
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	music_2
SELECT SUM ( T3.* ) , T1.Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Version_Number	book_press
SELECT SUM ( T1.alt ) , T4.* from circuits as T1 join races as T2 on T1.circuitId = T2.circuitId join pitStops as T3 on T2.raceId = T3.raceId join lapTimes as T4 WHERE T3.lap > UNKNOWN_VALUE	music_2
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE ORDER BY LIMIT 1	music_2
SELECT series_name , Language from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	music_2
SELECT T3.name , T1.dept_name from department as T1 join course as T2 on T1.dept_name = T2.dept_name join instructor as T3 on T1.dept_name = T3.dept_name join student as T4 on T1.dept_name = T4.dept_name join takes as T5 on T4.ID = T5.ID join section as T6 on T5.course_id = T6.course_id WHERE T2.course_id > UNKNOWN_VALUE GROUP BY T6.time_slot_id	music_2
SELECT T1.dept_name , dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id GROUP BY T4.semester LIMIT 1	book_press
SELECT T2.Jul from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID WHERE T1.Hanzi > UNKNOWN_VALUE	music_2
SELECT T3.Press_ID from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID GROUP BY T1.Author_ID ORDER BY COUNT ( Author_ID ) LIMIT 1	music_2
SELECT Code from Movies WHERE Title = UNKNOWN_VALUE	music_2
SELECT SUM ( T2.Title ) , T3.* from press as T1 join book as T2 on T1.Press_ID = T2.Press_ID join book as T3 GROUP BY T1.Press_ID	book_press
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id ORDER BY T3.justice_score	country_language
SELECT dept_name from department	country_language
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	country_language
SELECT organisation_id from Grants	country_language
SELECT SUM ( * ) from Order_Items	country_language
SELECT T1.Country , T2.Manager from country as T1 join team as T2	college_2
SELECT T2.* from game as T1 join game_player as T2 GROUP BY T1.Units_sold_Millions	country_language
SELECT MIN ( T1.product_type_code ) from Products as T1 join Addresses as T2 WHERE T2.address_id = UNKNOWN_VALUE	country_language
SELECT T1.destination , destination from flight as T1 join certificate as T2 GROUP BY T1.departure_date ORDER BY COUNT ( T2.* ) LIMIT 1	country_language
SELECT T1.Id from customers as T1 join goods as T2 join receipts as T3 GROUP BY T2.Price ORDER BY COUNT ( T3.* ) LIMIT 1	country_language
SELECT SUM ( T3.* ) , T2.Name from Movies as T1 join MovieTheaters as T2 on T1.Code = T2.Movie join MovieTheaters as T3 GROUP BY T1.Rating	college_2
SELECT T1.series_name from TV_Channel as T1 join Cartoon as T2 GROUP BY T1.Language HAVING T2.*	country_language
SELECT T3.Claim_Status_Code from Staff as T1 join Claims_Processing as T2 on T1.Staff_ID = T2.Staff_ID join Claim_Headers as T3 on T2.Claim_ID = T3.Claim_Header_ID WHERE T1.Staff_Details = UNKNOWN_VALUE	country_language
SELECT T2.dept_name , dept_name from department as T1 join instructor as T2 on T1.dept_name = T2.dept_name join student as T3 on T1.dept_name = T3.dept_name join takes as T4 on T3.ID = T4.ID join section as T5 on T4.course_id = T5.course_id GROUP BY T5.year HAVING dept_name	college_2
SELECT T1.Template_Type_Code , Template_Type_Code from Ref_Template_Types as T1 join Templates as T2 on T1.Template_Type_Code = T2.Template_Type_Code join Paragraphs as T3 GROUP BY T2.Version_Number ORDER BY COUNT ( T3.* ) LIMIT 1	country_language
SELECT SUM ( * ) from Tasks	country_language
SELECT SUM ( T2.Original_air_date ) , T3.* from TV_Channel as T1 join Cartoon as T2 on T1.id = T2.Channel join Cartoon as T3 GROUP BY T1.id	college_2
SELECT T2.Ordinal , T1.Food from goods as T1 join items as T2 on T1.Id = T2.Item join receipts as T3 ORDER BY COUNT ( T3.* ) LIMIT 1	country_language
SELECT MIN ( order_id ) from Orders	country_language
SELECT T1.economics_score from countries as T1 join official_languages as T2 ORDER BY COUNT ( T2.* )	country_language
SELECT aid , destination from flight WHERE aid < UNKNOWN_VALUE	country_language
SELECT SUM ( * ) from Order_Items	country_language
SELECT AVG ( Rating ) from TV_series	country_language
SELECT COUNT ( T2.* ) from Ref_Template_Types as T1 join Paragraphs as T2 WHERE T1.Template_Type_Code = UNKNOWN_VALUE	country_language
SELECT T3.Press_ID , T1.Name , T3.Month_Profits_billion from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID ORDER BY T2.Book_Series LIMIT 1	country_language
SELECT T2.organisation_id , T1.other_details from Documents as T1 join Grants as T2 on T1.grant_id = T2.grant_id join Organisations as T3 on T2.organisation_id = T3.organisation_id join Projects as T4 on T3.organisation_id = T4.organisation_id join Project_Outcomes as T5 on T4.project_id = T5.project_id join Tasks as T6 GROUP BY T5.project_id ORDER BY COUNT ( T6.* ) LIMIT 1	country_language
SELECT T1.Title , T3.AId from Songs as T1 join Tracklists as T2 on T1.SongId = T2.SongId join Albums as T3 on T2.AlbumId = T3.AId WHERE T3.Label = UNKNOWN_VALUE	country_language
SELECT T1.Customer_ID from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID WHERE T3.Claim_Header_ID like UNKNOWN_VALUE	country_language
SELECT AVG ( aid ) , aid from flight GROUP BY aid	college_2
SELECT T1.Id from customers as T1 join items as T2 WHERE T2.Ordinal > UNKNOWN_VALUE	country_language
SELECT T2.* from city as T1 join hosting_city as T2 WHERE T1.City = UNKNOWN_VALUE	country_language
SELECT SUM ( T1.Hanyu_Pinyin ) , T3.* from city as T1 join temperature as T2 on T1.City_ID = T2.City_ID join hosting_city as T3 GROUP BY T2.Aug	college_2
SELECT SUM ( * ) from Shipment_Items	country_language
SELECT Official_native_language from country	country_language
SELECT SUM ( T2.* ) from Orders as T1 join Shipment_Items as T2 GROUP BY T1.order_id	country_language
SELECT SUM ( T2.* ) from Songs as T1 join Vocals as T2 WHERE T1.Title = UNKNOWN_VALUE	country_language
SELECT T1.Customer_ID from Customers as T1 join Policies as T2 on T1.Customer_ID = T2.Customer_ID join Claim_Headers as T3 on T2.Policy_ID = T3.Policy_ID WHERE T3.Claim_Header_ID like UNKNOWN_VALUE	country_language
SELECT T1.parent_product_id , parent_product_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Shipment_Items as T4 WHERE T4.* > UNKNOWN_VALUE GROUP BY T3.order_id HAVING *	country_language
SELECT T1.id from languages as T1 join official_languages as T2 on T1.id = T2.language_id join countries as T3 on T2.country_id = T3.id join official_languages as T4 GROUP BY T3.politics_score ORDER BY COUNT ( T4.* ) LIMIT 1	country_language
SELECT T4.customer_id from Products as T1 join Order_Items as T2 on T1.product_id = T2.product_id join Orders as T3 on T2.order_id = T3.order_id join Customers as T4 on T3.customer_id = T4.customer_id join Shipment_Items as T5 on T2.order_item_id = T5.order_item_id join Shipments as T6 on T5.shipment_id = T6.shipment_id join Invoices as T7 on T6.invoice_number = T7.invoice_number join Shipment_Items as T8 WHERE T1.product_description > UNKNOWN_VALUE GROUP BY T7.invoice_date HAVING T8.*	country_language
SELECT organisation_id from Grants	country_language
SELECT Template_Type_Code , Template_Type_Code from Ref_Template_Types WHERE Template_Type_Description like UNKNOWN_VALUE	country_language
SELECT T4.Jul from city as T1 join hosting_city as T2 on T1.City_ID = T2.Host_City join match as T3 on T2.Match_ID = T3.Match_ID join temperature as T4 on T1.City_ID = T4.City_ID WHERE T1.City in UNKNOWN_VALUE ORDER BY T3.Result LIMIT 1	country_language
SELECT T1.dept_name from department as T1 join student as T2 on T1.dept_name = T2.dept_name join takes as T3 on T2.ID = T3.ID join section as T4 on T3.course_id = T4.course_id WHERE T4.semester > UNKNOWN_VALUE	country_language
SELECT SUM ( T3.* ) , T1.Candidate_ID from candidate as T1 join people as T2 on T1.People_ID = T2.People_ID join people as T3 WHERE T2.Sex < UNKNOWN_VALUE GROUP BY T1.Poll_Source	country_language
SELECT SUM ( * ) from Order_Items	country_language
SELECT SUM ( T2.* ) from races as T1 join lapTimes as T2 WHERE T1.raceId > UNKNOWN_VALUE	country_language
SELECT T1.LastName from customers as T1 join items as T2 WHERE T2.Ordinal > UNKNOWN_VALUE	country_language
SELECT Hight_definition_TV , Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	country_language
SELECT AVG ( T3.Year_Profits_billion ) from author as T1 join book as T2 on T1.Author_ID = T2.Author_ID join press as T3 on T2.Press_ID = T3.Press_ID WHERE T1.Name > UNKNOWN_VALUE GROUP BY Name	country_language
SELECT alt from circuits WHERE circuitRef > UNKNOWN_VALUE	country_language
SELECT Shop_ID from shop WHERE Location = UNKNOWN_VALUE	country_language
SELECT Id from customers	country_language
SELECT series_name from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	country_language
SELECT Hight_definition_TV from TV_Channel WHERE Hight_definition_TV = UNKNOWN_VALUE	country_language
SELECT SUM ( T1.eg Agree Objectives ) , T4.* from Tasks as T1 join Projects as T2 on T1.project_id = T2.project_id join Project_Outcomes as T3 on T2.project_id = T3.project_id join Tasks as T4 GROUP BY T3.project_id ORDER BY COUNT ( * ) LIMIT 1	country_language
SELECT MIN ( T2.alt ) , MIN ( T1.url ) from races as T1 join circuits as T2 on T1.circuitId = T2.circuitId join pitStops as T3 on T1.raceId = T3.raceId WHERE T3.lap = UNKNOWN_VALUE	country_language
SELECT T1.Id from customers as T1 join goods as T2 WHERE T2.Price > UNKNOWN_VALUE	country_language
SELECT SUM ( * ) from Shipment_Items	country_language
SELECT Title from Songs WHERE Title = UNKNOWN_VALUE	country_language
SELECT SUM ( * ) from Order_Items	country_language
SELECT SUM ( T2.Flavor ) , T1.Id from customers as T1 join goods as T2	college_2
SELECT T1.document_type_code from Document_Types as T1 join Documents as T2 on T1.document_type_code = T2.document_type_code join Grants as T3 on T2.grant_id = T3.grant_id join Organisations as T4 on T3.organisation_id = T4.organisation_id join Projects as T5 on T4.organisation_id = T5.organisation_id join Project_Outcomes as T6 on T5.project_id = T6.project_id join Research_Outcomes as T7 on T6.outcome_code = T7.outcome_code WHERE T7.outcome_description = UNKNOWN_VALUE	country_language
SELECT Id from customers WHERE FirstName = UNKNOWN_VALUE	country_language
SELECT T3.Shop_ID from device as T1 join stock as T2 on T1.Device_ID = T2.Device_ID join shop as T3 on T2.Shop_ID = T3.Shop_ID join stock as T4 WHERE T1.Device = UNKNOWN_VALUE GROUP BY Shop_ID ORDER BY COUNT ( T4.* ) LIMIT 1	device
SELECT T2.Title , T1.Platform_ID from platform as T1 join game as T2 on T1.Platform_ID = T2.Platform_ID	video_game
