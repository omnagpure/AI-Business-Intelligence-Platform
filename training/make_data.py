"""Build labelled column-name data for the optional semantic name model.

Design rules (so the evaluation means something):
* Seed names are written by hand, independently of semantic/column_lexicon.py.
* Seed names are split into train/test BEFORE any variants are generated, so no base name (and no
  case/prefix/suffix variant of it) is ever in both sets. The test set therefore measures
  generalisation to unseen names, not memorisation.
* Names vary the way real exports do: short forms (qty, amt, cust), underscores/camelCase/UPPER, prefixes
  (tbl_, raw_) and suffixes (_v2, _usd, _pct).
* A third set (UNSEEN_DOMAIN_NAMES) comes from industries absent from training, to test generalisation.
* A second, separate evaluation set is made of real column headers from public supply-chain
  datasets, labelled by hand (REAL_HEADERS). Those are never used for training.

All of this is synthetic or hand-labelled and covers names only. It is NOT a benchmark of real-world
accuracy. Run:  python -m training.make_data
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from semantic.column_lexicon import ROLES, tokenize

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
TEST_FRACTION = 0.3
VARIANTS_PER_BASE = 12

SEED_NAMES: dict[str, str] = {
    "revenue": "total sales, net revenue, gross sales, sales amount, order total, invoice amount, turnover, gmv, "
               "income, billing amount, revenue usd, sales value, amt net, txn amount, subscription revenue, mrr, "
               "arr, ad revenue, fees collected, receipts, gross merchandise value, booking value, sales inr, "
               "net sales, line total, order value, payment received, takings, proceeds, total billed, "
               "monthly recurring revenue, annual contract value, deal amount, cart value, sales per customer",
    "cost": "unit cost, cogs, total cost, expense, operating expense, opex, cost of goods sold, purchase cost, "
            "shipping cost, freight cost, marketing spend, ad spend, labor cost, material cost, overhead, "
            "expenditure, cost price, procurement cost, supplier cost, manufacturing cost, rent expense, "
            "salary expense, acquisition cost, spend usd, outgoing payments, cost per unit, handling fee, tax paid",
    "profit": "profit, gross profit, net profit, net income, ebitda, operating income, earnings, profit per order, "
              "benefit per order, contribution margin, profit amount, margin amount, operating profit, pbt, pat, "
              "net earnings, profit usd, gross margin value, order profit, trading profit, surplus, "
              "retained earnings, income after tax, profit loss, bottom line, earnings before tax, "
              "profit before tax, net gain",
    "price": "unit price, price, list price, mrp, msrp, selling price, product price, item price, price per unit, "
             "sale price, retail price, base price, price usd, discounted price, offer price, wholesale price, "
             "price per kg, ticket price, subscription price, plan price, average selling price, asp, "
             "monthly price, standard price, price inr, listing price, shelf price, avg unit price",
    "quantity": "quantity, qty, units sold, units, order quantity, items sold, number of items, stock level, "
                "inventory level, on hand, units produced, pieces, volume sold, qty ordered, quantity received, "
                "count of items, seats, headcount, number of units, shipped qty, defects, reorder point, "
                "demand forecast, units returned, qty shipped, no of orders, order count, line items, num units",
    "datetime": "date, order date, created at, updated at, timestamp, transaction date, invoice date, ship date, "
                "shipped on, delivery date, signup date, dt, event time, purchase date, due date, expected at, "
                "received at, posting date, hire date, start date, end date, last login, booking date, "
                "record date, month, period, week start, txn dt, created on, modified time, closing date",
    "identifier": "id, order id, transaction id, invoice number, uuid, record id, row id, sku id, warehouse id, "
                  "supplier id, machine id, run id, ticket id, case id, session id, order item id, "
                  "purchase order id, shipment id, tracking number, reference no, serial number, batch id, "
                  "employee id, lead id, account id, line id, txn ref, guid, index key, primary key, "
                  "voucher no, receipt no, department id, category id",
    "category": "category, segment, type, department name, status, channel, brand, tier, payment method, "
                "shipping mode, order status, delivery status, gender, plan type, subscription plan, "
                "customer segment, product category, department, class, grade, source, lead source, "
                "campaign type, payment type, marital status, job title, industry, device type, priority, "
                "size band, product status, mode of transport, carrier",
    "customer": "customer, customer id, client id, buyer, subscriber id, member id, user id, customer number, "
                "cust id, client name, account holder, customer ref, buyer id, shopper id, patron id, "
                "order customer id, cust no, user key, consumer id, client code, subscriber, member, user, "
                "account number, customer key, client ref, customer code, end user id, customer name, tenant id",
    "product": "product, product name, item name, sku, product id, article, item, product code, product title, "
               "merchandise, product sku, item id, goods, sku code, model name, product line, product number, "
               "stock keeping unit, item code, product key, catalog item, part number, part name, "
               "product card id, material name, article number, model number, product family, product item",
    "location": "city, state, country, region, zip code, postal code, address, street, latitude, longitude, lat, "
                "lng, location, territory, district, market, warehouse region, supplier country, store location, "
                "shipping city, destination, origin country, province, county, area, branch city, pincode, "
                "ship to state, billing country, order region, geo, site, zone, locality",
    "percentage": "percent, discount rate, profit ratio, conversion rate, churn rate, growth rate, margin pct, "
                  "tax rate, interest rate, share, utilization, ctr, click through rate, bounce rate, "
                  "attrition rate, discount percent, fill rate, defect rate, success ratio, occupancy, "
                  "yield pct, completion rate, open rate, return rate, win rate, retention rate, pct change, "
                  "proportion, cagr, uptime percentage",
    "balance": "balance, account balance, closing balance, opening balance, outstanding amount, current balance, "
               "ledger balance, equity, deposit, cash balance, available balance, loan balance, credit balance, "
               "debit balance, net worth, wallet balance, outstanding, due balance, principal outstanding, "
               "savings, bank balance, remaining balance, overdraft, liabilities, assets, receivable balance, "
               "payable balance, balance usd, carrying value",
    "duration": "duration, days for shipping, lead time days, tenure, age, hours worked, delivery days, "
                "processing time, downtime minutes, time to resolve, response time, session duration, "
                "days to ship, days since signup, wait time, cycle time, shipping delay, length of stay, "
                "experience years, minutes spent, handle time, turnaround time, days overdue, "
                "resolution hours, supplier lead time, tenure months, age in days, elapsed seconds, "
                "days scheduled",
    "target": "churn, churned, is fraud, fraud flag, default, late delivery risk, converted, outcome, label, "
              "target, response, will buy, attrition, is churned, stockout flag, defaulted, readmitted, "
              "survived, clicked, purchased, class label, risk flag, late, is late, cancelled, returned flag, "
              "y, positive outcome, subscribed, has left, loan default, delayed",
    "text": "description, comments, review, notes, feedback, remarks, message, summary, product description, "
            "customer review, ticket text, complaint text, email body, comment text, reviewer comments, "
            "details, narrative, bio, abstract, free text, additional info, review text, support notes, "
            "address notes, job description, memo, explanation, observation, caption, tweet text",
    "unknown": "x, x1, x2, var1, var 2, col3, field a, field b, unnamed 0, unnamed 4, misc, other, tmp, temp, "
               "aux, z, f1, f2, attr7, column 12, data, value1, val, metric a, param, q, k, extra, spare, "
               "zz, abc, flag3, v10, unnamed 12",
}


# More names, written to cover more industries and the messy styles seen in real exports.
SEED_NAMES_EXTRA: dict[str, str] = {
    "revenue": "gross revenue, sales revenue, total income, rental income, tuition fees, license revenue, commission earned, "
               "premium written, royalty income, service revenue, product sales, online sales, retail sales, wholesale sales, "
               "store sales, daily sales, monthly sales, quarterly revenue, revenue per user, avg order value, aov, basket value, "
               "ticket revenue, donation amount, fare collected, toll collected, premiums, interest income, dividend income, "
               "grant received, net billings, bookings, invoice total, bill amount, collections, gross receipts, "
               "sales before tax, sales after discount, ordered amount, revenue ytd, revenue mtd, arpu",
    "cost": "total expenses, direct cost, indirect cost, fixed cost, variable cost, production cost, storage cost, warehousing cost, "
            "transport cost, logistics cost, packaging cost, customer acquisition cost, cac, cost to serve, cost of sales, "
            "cost of revenue, input cost, raw material cost, energy cost, fuel cost, maintenance cost, repair expense, "
            "insurance expense, travel expense, payroll expense, wages paid, salary cost, vendor payment, supplier payment, "
            "rent paid, utilities expense, depreciation, amortization, interest expense, bad debt expense, marketing cost, "
            "advertising spend, promo spend, shipping charge, freight charges, landed cost, standard cost, avg cost, cost usd",
    "profit": "gross margin, net margin amount, operating margin value, profit margin amount, profit after tax, profit before interest, "
              "net operating income, noi, ebit, operating earnings, net surplus, net result, income from operations, "
              "order profit per item, profit per unit, unit profit, line profit, gross profit usd, net profit inr, profit ytd, "
              "profit mtd, profit loss amount, net profit after tax, earning, total profit, total earnings, margin per order, "
              "contribution, gain loss, realised gain, pnl, profit contribution, profit per customer, markup amount",
    "price": "price each, unit selling price, unit list price, price before tax, price after discount, net price, gross price, "
             "special price, promo price, sale price per unit, rate per unit, rate per kg, rate per hour, hourly rate, daily rate, "
             "room rate, fare, ticket fare, tariff price, price per sqft, price per item, price per piece, price per ton, "
             "price per liter, price per share, share price, stock price, closing price, opening price, bid price, ask price, "
             "vendor price, contract price, floor price, ceiling price, price eur, price gbp, price local, current price, "
             "new price, old price, avg price, product mrp, item mrp, pack price",
    "quantity": "units shipped, units received, units ordered, units in stock, stock on hand, stock quantity, closing stock, "
                "opening stock, quantity on hand, qty on hand, available quantity, available stock, backorder qty, safety stock, "
                "min stock, max stock, reorder qty, order qty, ordered qty, received qty, returned qty, damaged units, "
                "defective units, scrap qty, produced qty, planned qty, actual qty, demand units, forecast units, forecast quantity, "
                "sales volume, volume units, number of orders, number of customers, number of employees, passenger count, "
                "visitor count, ticket count, page views, impressions, clicks, sessions count, items per order, lines per order, "
                "cases, pallets, boxes shipped, cartons, tonnage",
    "datetime": "order timestamp, order datetime, order time, created datetime, created timestamp, creation date, modified date, "
                "last modified, last updated, update date, updated on, insert date, load date, etl date, extract date, "
                "snapshot date, report date, reporting date, as of date, effective date, expiry date, expiration date, "
                "maturity date, issue date, invoice dt, bill date, payment date, paid on, settlement date, ship dt, dispatch date, "
                "dispatched at, delivered at, arrival date, departure date, check in date, check out date, joined on, join date, "
                "registration date, signup ts, first order date, last order date, event date, event ts, trans date, "
                "txn timestamp, posting dt, value date, fiscal year, fiscal month, fiscal quarter, calendar date, week ending, "
                "month end, year month, yyyymm, dt key, date key, scheduled date, planned date, actual date, promised date, "
                "closed date, resolved at, opened at, start time, end time, begin date, appointment date, admission date, "
                "discharge date, enrollment date, graduation date, harvest date, launch date",
    "identifier": "order no, order number, order ref, order key, order item key, invoice id, invoice no, bill no, bill id, "
                  "payment id, payment ref, txn no, trans id, transaction ref, transaction key, shipment no, consignment no, "
                  "awb number, waybill no, tracking id, package id, container id, pallet id, lot number, batch no, batch number, "
                  "serial no, serial id, asset id, asset tag, device id, imei, vehicle id, vin, plate number, route id, trip id, "
                  "flight number, booking id, booking ref, reservation id, ticket no, ticket number, case number, claim id, "
                  "claim no, policy number, policy id, loan id, loan number, account id, acct no, acct id, patient id, "
                  "student id, roll no, employee no, emp id, emp code, staff id, badge no, store id, store code, branch id, "
                  "branch code, outlet id, site id, vendor id, vendor code, supplier code, supplier no, warehouse code, wh id, "
                  "location id, region id, campaign id, ad id, adgroup id, session key, visitor id, cookie id, event id, log id, "
                  "request id, trace id, row number, row no, serial, seq no, sequence number, surrogate key, natural key, hash key",
    "category": "product group, product type, product class, item category, item type, item group, item class, sub category, "
                "subcategory, category level 1, category l2, main category, parent category, dept name, division, business unit, "
                "bu, vertical, sector, sub sector, industry type, customer type, customer class, customer tier, customer group, "
                "client type, account type, account status, order type, order channel, sales channel, order priority, "
                "ticket priority, severity, risk level, risk rating, credit rating, credit grade, rating band, age group, "
                "age band, income bracket, income band, salary grade, pay grade, job level, job family, employment type, "
                "contract type, work mode, shift, shift type, marital, education level, qualification, blood group, "
                "diagnosis category, payment status, delivery type, delivery method, shipping method, ship mode, shipping carrier, "
                "transport mode, fulfillment type, return reason, cancel reason, source channel, traffic source, utm source, "
                "utm medium, referral source, device category, browser, operating system, os, platform, app version, "
                "subscription tier, plan name, pricing plan, license type, membership level, loyalty tier, color, colour, size, "
                "material type, packaging type, unit of measure, uom, currency, currency code, language, season, weekday, "
                "day of week, month name, quarter",
    "customer": "customer name, cust name, client number, client key, client ref no, customer account, customer account id, "
                "customer no, cust code, cust key, cust ref, cust number, customer uid, customer uuid, buyer name, buyer key, "
                "shopper key, purchaser id, purchaser, subscriber key, subscriber number, subscriber ref, member number, member no, "
                "member key, membership id, member code, user uid, user guid, user ref, userid, uid, account holder id, "
                "policy holder id, patient, tenant key, company id, company name, org id, organisation id, organization name, "
                "business id, merchant id, merchant name, partner id, reseller id, distributor id, dealer id, dealer code, agent id, "
                "contact id, prospect id, household id, loyalty id, loyalty number, card holder id, wallet user id, guest id, "
                "passenger id, rider id, borrower id, depositor id, payer id",
    "product": "product label, product alias, item label, item title, item number, item no, item key, sku number, sku no, sku key, "
               "sku name, upc, ean, gtin, barcode, isbn, asin, part no, part id, part code, component name, component id, "
               "material id, material code, material number, material no, ingredient, ingredient name, spare part, spare part no, "
               "model code, model no, model id, model sku, variant name, variant id, variant sku, style code, style number, "
               "style name, article code, article id, article name, catalog number, catalogue id, listing id, listing title, "
               "offer id, bundle name, bundle id, drug name, medicine name, book title, course name, service name, service code, "
               "dish name, menu item, vehicle model, make model, device model",
    "location": "city name, town, village, state name, state code, province name, country name, country code, region name, "
                "region code, continent, county name, zip, zipcode, zip5, postcode, post code, pin code, postal, address line 1, "
                "address line 2, address1, addr, street address, street name, billing city, billing state, billing country, "
                "billing zip, shipping state, shipping country, shipping address, shipping zip, ship to city, ship to country, "
                "ship from, ship from city, delivery city, delivery zip, delivery address, destination city, destination country, "
                "destination state, origin city, origin state, origin region, pickup location, pickup city, drop location, "
                "dropoff city, store city, store state, store region, branch location, branch region, branch address, "
                "warehouse city, warehouse location, warehouse country, plant location, plant city, supplier city, supplier state, "
                "supplier region, vendor country, vendor city, customer country, customer region, customer address, lat long, "
                "latitude deg, longitude deg, geo lat, geo lon, gps lat, gps lng, coordinates, geolocation, geo hash, "
                "sales region, sales territory, sales area, market region, area code, zone name, zone code, district name, "
                "locality name, neighborhood, suburb, ward, taluk, tehsil, mandal, locale",
    "percentage": "pct, percent complete, percentage change, percent change, percent of total, pct of total, share of total, "
                  "market share, share pct, discount pct, discount percentage, disc rate, disc pct, markup pct, margin percent, "
                  "margin percentage, profit margin pct, gross margin pct, net margin pct, operating margin pct, roi pct, roi, "
                  "return on investment, interest pct, apr, apy, rate of return, growth pct, yoy growth, mom growth, qoq growth, "
                  "yoy change pct, churn pct, retention pct, attrition pct, turnover rate, absenteeism rate, utilisation rate, "
                  "utilization pct, capacity utilization, occupancy rate, fill pct, service level, on time rate, "
                  "on time delivery pct, otd pct, defect pct, scrap rate, rework rate, yield rate, failure rate, error rate, "
                  "return pct, refund rate, cancellation rate, bounce pct, ctr pct, click rate, open pct, conversion pct, win pct, "
                  "completion pct, progress pct, attendance pct, pass rate, pass percentage, score pct, tax pct, gst rate, "
                  "vat rate, tax percent, commission rate, commission pct, probability, prob, likelihood pct, confidence pct, "
                  "ratio, debt ratio, current ratio, quick ratio",
    "balance": "balance amount, balance due, amount due, amount outstanding, outstanding balance, outstanding principal, "
               "principal balance, remaining principal, closing bal, opening bal, bal, balance brought forward, "
               "balance carried forward, balance bf, balance cf, ending balance, beginning balance, starting balance, "
               "running balance, available credit, credit available, available funds, account bal, acct balance, savings balance, "
               "checking balance, current account balance, fd balance, deposit balance, loan outstanding, loan principal, "
               "card balance, wallet bal, points balance, loyalty points balance, leave balance, pto balance, vacation balance, "
               "stock balance, inventory balance, ledger bal, gl balance, trial balance, cash in hand, cash at bank, bank bal, "
               "net balance, total balance, balance after txn, balance after transaction, post balance, pre balance, "
               "receivables, payables, accounts receivable, accounts payable, ar balance, ap balance, total assets, "
               "total liabilities, total equity, shareholders equity, owners equity, working capital, net assets",
    "duration": "duration minutes, duration seconds, duration hours, duration days, duration sec, duration mins, call duration, "
                "talk time, hold time, ring time, wait duration, waiting time, queue time, lead time, lead time hrs, "
                "lead time weeks, transit time, transit days, delivery time, delivery lead time, delivery delay days, "
                "shipping time, ship days, days to deliver, days to close, days to resolve, days open, days pending, "
                "days outstanding, days past due, days late, delay minutes, delay hours, late by days, overdue days, age days, "
                "age months, age years, account age, customer tenure, tenure days, tenure years, employee tenure, "
                "length of service, years of service, service years, months active, months on book, time on site, time spent, "
                "time taken, time to hire, time to first response, first response time, resolution time, ttr, mttr, mtbf, "
                "uptime hours, downtime hours, runtime, run time, operating hours, working hours, hours per week, "
                "overtime hours, shift length, trip duration, travel time, commute time, ride duration, flight duration, "
                "stay duration, session length, video length, watch time, play time, cycle time days, takt time, setup time, "
                "processing days, turnaround days, sla days, sla hours, warranty period, warranty months, contract length, "
                "contract months, notice period, probation period, elapsed days, elapsed time, time since last order, "
                "recency days, recency",
    "target": "is churn, churn flag, churn label, churn status, will churn, did churn, left company, has churned, "
              "cancelled flag, cancelled subscription, subscription cancelled, is fraudulent, fraud label, fraud indicator, "
              "is fraud transaction, suspicious flag, default flag, default status, loan defaulted, is default, bad loan, "
              "delinquent, is delinquent, npa flag, readmission, readmitted flag, is readmitted, survival, survived flag, "
              "mortality, is dead, death flag, diagnosis positive, disease flag, outcome label, outcome flag, result label, "
              "class target, target variable, target value, target class, target label, dependent variable, label y, y label, "
              "y target, y true, ground truth, truth label, converted flag, is converted, conversion flag, did convert, "
              "purchased flag, bought, did buy, will purchase, clicked flag, is click, did click, opened flag, responded, "
              "response flag, responded flag, accepted offer, offer accepted, approved, is approved, late flag, "
              "is late delivery, delivered late, late shipment, delayed flag, is delayed, stockout, stock out flag, "
              "out of stock flag, is stockout, defect flag, is defective, has defect, passed qc, failed flag, pass fail, "
              "success flag, is success, is successful, win flag, won, lost flag, is won, is lost, is returned, return flag, "
              "high risk flag, risk label, at risk, is at risk, attrition flag, resigned, has resigned",
    "text": "product details, item description, long description, short description, detailed description, full description, "
            "description text, desc, remarks text, remark, comment, user comment, customer comment, customer feedback, "
            "customer notes, feedback text, feedback comment, survey response, open ended response, free text response, "
            "review body, review title, review content, review comment, rating comment, complaint, complaint description, "
            "issue description, problem description, problem statement, incident description, incident summary, "
            "ticket description, ticket subject, ticket body, ticket comments, support comment, agent notes, agent comments, "
            "call notes, call summary, meeting notes, visit notes, internal notes, private notes, admin notes, order notes, "
            "delivery notes, shipping notes, special instructions, instructions, delivery instructions, handling instructions, "
            "gift message, message body, message text, email subject, email text, sms text, chat transcript, transcript, "
            "conversation text, tweet, post text, post content, caption text, article text, article body, headline, title text, "
            "bio text, profile bio, clinical notes, doctor notes, diagnosis notes, discharge summary, lab remarks, "
            "inspection notes, inspection remarks, qc comments, quality notes, maintenance notes, work description, "
            "task description, job summary, job details, requirements text, resume text, cover letter, reason text, "
            "justification, explanation text, details text",
    "unknown": "x3, x4, x5, x10, var3, var 4, var5, variable 1, variable 2, v1, v2, v3, v4, col1, col2, col 4, col 5, column1, "
               "column 2, column 3, field1, field 2, field 3, field c, field d, attr1, attr 2, attribute 3, feature1, feature 2, "
               "feat3, f3, f4, f5, p1, p2, param1, parameter 2, c1, c2, c3, a, b, c, d, e, aa, bb, cc, xx, yy, abc1, test, test1, "
               "test col, dummy, dummy1, dummy col, tmp1, tmp col, temp1, temp col, temp value, placeholder, placeholder 1, "
               "unnamed 1, unnamed 2, unnamed 3, unnamed 5, unnamed 7, unnamed 10, index, idx, level 0, level 1, sheet1, "
               "sheet col, blank, blank col, empty, empty col, none, other1, other 2, misc1, misc 2, misc col, misc data, "
               "misc info, extra1, extra 2, extra col, spare1, spare 2, spare col, aux1, aux 2, aux col, reserved, reserved1, "
               "reserved field, flag, flag1, flag 2, ind, indicator, indicator1, code1, code 2, val1, val 2, value 2, value3, "
               "value col, data1, data 2, data col, info, info1, info2, number1, num1, num 2, no1, stuff, thing, things, "
               "metric1, metric 2, metric b, measure1, measure 2, measure b, score, score1, score 2, rank, rank1",
}
for _role, _more in SEED_NAMES_EXTRA.items():
    SEED_NAMES[_role] = SEED_NAMES[_role] + ", " + _more

# Supply-chain roles (hand-written, independent of semantic/column_lexicon.py).
SEED_NAMES["supplier"] = (
    "supplier, supplier name, supplier id, vendor, vendor name, vendor id, vendor code, supplier code, manufacturer, "
    "manufacturer name, supplier company, preferred supplier, primary vendor, vendor account, supplier no, "
    "sourcing partner, contract supplier, supplier key, vendor key, procurement vendor")
SEED_NAMES["warehouse"] = (
    "warehouse, warehouse id, warehouse name, warehouse code, depot, depot id, distribution centre, distribution center, "
    "dc id, fulfilment centre, fulfillment center, plant, plant id, facility, facility id, storage location, hub id, "
    "stocking point, site name, factory id, wh code")
SEED_NAMES["flag"] = (
    "on time delivery, is on time, delivered on time, in full, otif flag, stockout, stockout flag, stock out, is delayed, "
    "delayed flag, was late, is late, backorder flag, backordered, promotion flag, promo flag, on promotion, "
    "is stockout, delivered in full, late shipment flag, expedited flag, is expedited, quality hold flag, rejected flag")

# Names from industries that are NOT in the training seeds. Used only to test generalisation to new domains
# (hospital, school, farm, telecom, property, HR). Never used for training. Bases that collide with a
# training base are dropped at build time.
UNSEEN_DOMAIN_NAMES: dict[str, str] = {
    "revenue": "tuition collected, premium income, rental revenue, room revenue, call revenue, data revenue, crop sales, energy sales, ticket sales",
    "cost": "drug cost, treatment cost, equipment cost, seed cost, fertilizer cost, fuel expense, tower maintenance cost, training cost, recruitment cost, claims paid",
    "profit": "operating surplus, hospital surplus, net operating profit, plant profit, project profit, net farm income, profit per acre, profit per bed",
    "price": "room tariff price, rent per sqft, rent price, tuition price, crop price, mandi price, power tariff price, price per acre, price per bed, price per mb, price per minute, price per kwh",
    "quantity": "beds occupied, patients admitted, students enrolled, enrolment count, crop yield kg, acres planted, kwh consumed, units consumed, sim count, subscribers count, vacant units, vacancies",
    "datetime": "admission ts, enrollment dt, harvest dt, billing cycle date, lease start date, lease end date, hire dt, termination date, appointment ts, activation date, recharge date, meter reading date, sowing date, exam date",
    "identifier": "patient no, mrn, student no, enrollment no, farm id, plot id, parcel id, meter id, meter no, sim id, tower id, cell id, lease id, unit no, property id, listing no, applicant id, requisition id, candidate id",
    "category": "ward type, bed type, diagnosis group, specialty, course type, grade level, subject, crop type, soil type, irrigation type, tariff plan, handset type, network type, property type, unit type, furnishing status, employment status, hiring source, performance rating, designation, cost center",
    "customer": "patient name, student name, farmer id, farmer name, tenant, tenant name, resident id, borrower, policyholder id, applicant name, candidate name, guardian id",
    "product": "drug code, procedure name, treatment name, course title, subject name, crop name, variety name, seed variety, handset model, device make, property name, project name, scheme name, insurance product",
    "location": "hospital city, campus location, school district, village name, mandal name, farm location, plot location, cell site location, circle, telecom circle, tower location, property city, project location, branch pincode, catchment area",
    "percentage": "occupancy pct, bed occupancy rate, pass pct, attendance rate, dropout rate, enrolment ratio, crop yield pct, moisture pct, humidity pct, network uptime pct, call drop rate, vacancy rate, rent growth pct, absenteeism pct, offer acceptance rate",
    "balance": "patient balance, student fee balance, fee due balance, rent outstanding, security deposit, deposit held, prepaid balance, postpaid outstanding, meter credit balance, escrow balance, advance balance, loan outstanding amount",
    "duration": "length of stay days, treatment duration, class duration, course length hours, growing season days, irrigation hours, outage minutes, outage duration, network downtime mins, vacancy days, days on market, time to fill, time to hire days, notice days, probation months, experience months",
    "target": "readmitted within 30 days, no show, is no show, dropped out, dropout flag, passed exam, pass flag, crop failed, is spoiled, outage flag, port out flag, lease renewed, renewal flag, left job, is attrited, offer accepted flag, hired flag",
    "text": "doctor remarks, clinical remarks, teacher remarks, report card comment, farm notes, field observation, technician notes, outage description, property description, listing description, agent remarks, interview feedback, performance review text, exit interview notes, resume summary",
    "unknown": "zz1, v7, v8, col9, col 10, field e, field f, tmpcol, misc 3, extra 4, unnamed 15, unnamed 20, x9, x11, attr9, column 15",
}

# Hand-labelled headers from the public supply-chain files used while building this project.
# Ambiguous columns (for example "Order Item Discount", "SKU_ID") and personal-data columns are left out
# on purpose. These labels are one person's judgement, not ground truth. Never used for training.
REAL_HEADERS: dict[str, str] = {
    # DataCo supply chain
    "Type": "category", "Days for shipping (real)": "duration", "Days for shipment (scheduled)": "duration",
    "Benefit per order": "profit", "Sales per customer": "revenue", "Delivery Status": "category",
    "Late_delivery_risk": "target", "Category Id": "identifier", "Category Name": "category",
    "Customer City": "location", "Customer Country": "location", "Customer Id": "customer",
    "Customer Segment": "category", "Customer State": "location", "Customer Zipcode": "location",
    "Department Id": "identifier", "Department Name": "category", "Latitude": "location",
    "Longitude": "location", "Market": "location", "Order City": "location", "Order Country": "location",
    "Order Customer Id": "customer", "order date (DateOrders)": "datetime", "Order Id": "identifier",
    "Order Item Cardprod Id": "identifier", "Order Item Discount Rate": "percentage",
    "Order Item Id": "identifier", "Order Item Product Price": "price", "Order Item Profit Ratio": "percentage",
    "Order Item Quantity": "quantity", "Sales": "revenue", "Order Item Total": "revenue",
    "Order Profit Per Order": "profit", "Order Region": "location", "Order State": "location",
    "Order Status": "category", "Product Card Id": "identifier", "Product Category Id": "identifier",
    "Product Description": "text", "Product Name": "product", "Product Price": "price",
    "Product Status": "category", "shipping date (DateOrders)": "datetime", "Shipping Mode": "category",
    # supply_chain_dataset1
    "Date": "datetime", "Warehouse_ID": "warehouse", "Supplier_ID": "supplier", "Region": "location",
    "Units_Sold": "quantity", "Inventory_Level": "quantity", "Supplier_Lead_Time_Days": "duration",
    "Reorder_Point": "quantity", "Order_Quantity": "quantity", "Unit_Cost": "cost", "Unit_Price": "price",
    "Demand_Forecast": "quantity",
    # manufacturing sample
    "run_id": "identifier", "product_sku": "product", "machine_id": "identifier",
    "units_produced": "quantity", "defects": "quantity", "downtime_minutes": "duration",
    # inventory workbook
    "purchase_order_id": "identifier", "ordered_at": "datetime", "expected_at": "datetime",
    "received_at": "datetime", "status": "category", "supplier_country": "location",
    "warehouse_region": "location", "line_count": "quantity", "quantity_ordered": "quantity",
    "quantity_received": "quantity",
}

# ---- realistic name variation ------------------------------------------------------------------
# Real exports use short forms, underscores, prefixes and suffixes. Most affixes carry no role
# information (so the model learns to ignore them); a few (_usd, _pct, _dt, _id ...) do, as in real data.
_PREFIXES = ["src", "raw", "final", "tbl", "fld", "col", "stg", "dim", "fact", "ext", "int", "tmp", "old", "new", "cur", "x"]
_NEUTRAL_SUFFIXES = ["v2", "1", "2", "old", "new", "clean", "copy", "tmp", "bkp", "final", "raw", "x", "y", "2023",
                     "fy24", "adj", "calc", "orig", "ext"]
_ROLE_SUFFIXES: dict[str, list[str]] = {
    "revenue": ["usd", "inr", "eur", "amt", "val", "ytd", "mtd"], "cost": ["usd", "inr", "eur", "amt", "val", "ytd"],
    "profit": ["usd", "inr", "eur", "amt", "val", "ytd"], "price": ["usd", "inr", "eur", "amt", "val"],
    "balance": ["usd", "inr", "eur", "amt", "val"], "quantity": ["qty", "cnt", "count", "units", "nos"],
    "percentage": ["pct", "percent", "rate", "ratio"], "datetime": ["dt", "ts", "date", "time", "at"],
    "identifier": ["id", "no", "num", "key", "code"], "duration": ["days", "hrs", "mins", "hours", "min", "sec"],
    "location": ["name", "code"], "category": ["name", "type", "cd"], "customer": ["no", "name", "key"],
    "product": ["name", "code", "title"], "target": ["flag", "ind", "yn"], "text": ["text", "txt"],
}
_ABBR: dict[str, list[str]] = {
    "quantity": ["qty", "quan"], "amount": ["amt"], "customer": ["cust", "cstmr"], "number": ["no", "num", "nbr"],
    "transaction": ["txn", "trans"], "date": ["dt"], "identifier": ["id"], "price": ["prc"], "revenue": ["rev"],
    "order": ["ord"], "product": ["prod"], "category": ["cat", "categ"], "department": ["dept"],
    "description": ["desc"], "percent": ["pct"], "percentage": ["pct"], "balance": ["bal"], "duration": ["dur"],
    "discount": ["disc"], "supplier": ["supp", "sup"], "warehouse": ["wh", "whse"], "inventory": ["invt"],
    "employee": ["emp"], "total": ["tot", "ttl"], "average": ["avg"], "minimum": ["min"], "maximum": ["max"],
    "status": ["sts", "stat"], "shipping": ["shp", "ship"], "delivery": ["dlvry"], "payment": ["pmt"],
    "invoice": ["invc"], "reference": ["ref"], "account": ["acct", "acc"], "address": ["addr"],
    "country": ["cntry"], "region": ["rgn", "reg"], "latitude": ["lat"], "longitude": ["lon", "lng"],
    "month": ["mth", "mon"], "year": ["yr"], "week": ["wk"], "hours": ["hrs"], "minutes": ["mins"], "days": ["dys"],
    "location": ["loc"], "quarter": ["qtr"], "subscription": ["subs"], "expense": ["exp"], "cost": ["cst"],
    "profit": ["prft"], "margin": ["mrgn"], "forecast": ["fcst"], "purchase": ["purch"], "sales": ["sls"],
    "units": ["u", "unts"], "item": ["itm"], "unit": ["unt"], "stock": ["stk"], "received": ["rcvd"],
    "ordered": ["ordd"], "shipped": ["shpd"], "created": ["crtd"], "updated": ["upd"], "current": ["curr"],
}
_STYLES = ["snake", "camel", "pascal", "upper", "title_snake", "kebab", "title", "lower_space", "concat", "dot", "upper_concat"]
_STYLE_WEIGHTS = [30, 10, 8, 14, 6, 3, 10, 7, 5, 4, 3]


def normalise(name: str) -> str:
    return " ".join(tokenize(name))


def _render(tokens: list[str], style: str) -> str:
    if style == "snake":
        return "_".join(tokens)
    if style == "upper":
        return "_".join(tokens).upper()
    if style == "kebab":
        return "-".join(tokens)
    if style == "dot":
        return ".".join(tokens)
    if style == "title_snake":
        return "_".join(t.capitalize() for t in tokens)
    if style == "title":
        return " ".join(t.capitalize() for t in tokens)
    if style == "lower_space":
        return " ".join(tokens)
    if style == "camel":
        return tokens[0] + "".join(t.capitalize() for t in tokens[1:])
    if style == "pascal":
        return "".join(t.capitalize() for t in tokens)
    if style == "upper_concat":
        return "".join(tokens).upper()
    return "".join(tokens)       # concat


def _abbreviate(tokens: list[str], rng: random.Random) -> list[str]:
    out = [rng.choice(_ABBR[t]) if t in _ABBR and rng.random() < 0.7 else t for t in tokens]
    return out


def _variants(tokens: list[str], rng: random.Random, n: int, role: str = "unknown") -> list[str]:
    """Spelling variants of one base name: casing, separators, short forms, prefixes and suffixes."""
    out: list[str] = []
    for i in range(n):
        toks = list(tokens)
        if i < 2:                                   # keep two clean variants per base name
            style = ("snake", "camel")[i]
        else:
            if rng.random() < 0.45:
                toks = _abbreviate(toks, rng)
            if rng.random() < 0.35:
                toks = [rng.choice(_PREFIXES)] + toks
            suffixes = _ROLE_SUFFIXES.get(role)
            if suffixes and rng.random() < 0.25:
                suf = rng.choice(suffixes)
                if suf != toks[-1]:
                    toks.append(suf)
            if rng.random() < 0.35:
                toks.append(rng.choice(_NEUTRAL_SUFFIXES))
            style = rng.choices(_STYLES, weights=_STYLE_WEIGHTS)[0]
        out.append(_render(toks, style))
    return list(dict.fromkeys(out))


def _bases(source: dict[str, str] | None = None) -> list[tuple[str, str, tuple[str, ...]]]:
    """(role, base name, token key). Names whose token set appears under two roles are dropped as ambiguous;
    repeated token sets within a role are kept once (so a base can never straddle the train/test split)."""
    source = SEED_NAMES if source is None else source
    roles_of: dict[tuple[str, ...], set[str]] = {}
    first: dict[tuple[str, ...], tuple[str, str]] = {}
    for role, text in source.items():
        for raw in (x.strip() for x in text.split(",")):
            if not raw:
                continue
            key = tuple(sorted(set(tokenize(raw))))
            roles_of.setdefault(key, set()).add(role)
            first.setdefault(key, (role, raw))
    return [(r, b, k) for k, (r, b) in first.items() if len(roles_of[k]) == 1 and r in ROLES]


def build_datasets(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) frames with columns name, role, base. Disjoint by base name."""
    rng = random.Random(seed)
    train_rows, test_rows = [], []
    by_role: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for role, base, key in _bases():
        by_role.setdefault(role, []).append((base, key))
    for role in sorted(by_role):
        bases = sorted(set(by_role[role]))
        rng.shuffle(bases)
        n_test = max(1, round(len(bases) * TEST_FRACTION))
        for i, (base, _) in enumerate(bases):
            target = test_rows if i < n_test else train_rows
            for v in _variants(tokenize(base), rng, VARIANTS_PER_BASE, role):
                target.append({"name": v, "role": role, "base": base})
    train, test = pd.DataFrame(train_rows), pd.DataFrame(test_rows)
    leak = set(train["base"]) & set(test["base"])
    if leak:                                           # cannot happen by construction; guard anyway
        raise RuntimeError(f"Train/test leakage on base names: {sorted(leak)[:5]}")
    return train, test


def build_unseen_domain(seed: int = SEED) -> pd.DataFrame:
    """Names from industries absent from training (see UNSEEN_DOMAIN_NAMES), for a harder generalisation test."""
    seen_keys = {k for _, _, k in _bases()}
    rng = random.Random(seed + 1)
    rows = []
    for role, base, key in sorted(_bases(UNSEEN_DOMAIN_NAMES)):
        if key in seen_keys:
            continue
        for v in _variants(tokenize(base), rng, 8, role):
            rows.append({"name": v, "role": role, "base": base})
    return pd.DataFrame(rows)


def real_headers_frame() -> pd.DataFrame:
    return pd.DataFrame({"name": list(REAL_HEADERS), "role": list(REAL_HEADERS.values())})


def label_distribution(df: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(df["role"]).items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=DATA_DIR)
    a = p.parse_args()
    train, test = build_datasets(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    train.to_csv(a.out / "train.csv", index=False)
    test.to_csv(a.out / "test.csv", index=False)
    real_headers_frame().to_csv(a.out / "real_headers.csv", index=False)
    unseen = build_unseen_domain(a.seed)
    unseen.to_csv(a.out / "unseen_domain.csv", index=False)
    print(f"train: {len(train)} rows from {train['base'].nunique()} base names")
    print(f"test : {len(test)} rows from {test['base'].nunique()} base names (disjoint from train)")
    print(f"real : {len(REAL_HEADERS)} hand-labelled headers (evaluation only)")
    print(f"unseen domains: {len(unseen)} rows from {unseen['base'].nunique()} base names (evaluation only)")
    print("train label counts:", label_distribution(train))
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
