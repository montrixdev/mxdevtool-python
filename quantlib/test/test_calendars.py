"""
 Copyright (C) 2023 Skandinaviska Enskilda Banken AB (publ)

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
"""
import itertools
import unittest

import mxdevtool as ql


class JointCalendarTest(unittest.TestCase):

    def test_joint_calendar_holidays(self):
        base_calendars = [ql.Sweden(), ql.Denmark(), ql.Finland(), ql.Norway(), ql.Iceland()]
        joint_nordics = ql.JointCalendar(base_calendars)
        start_date = ql.Date(1, ql.January, 2023)
        end_date = ql.Date(31, ql.December, 2023)

        joint_holidays = set(joint_nordics.holidayList(start_date, end_date))
        base_holidays = [calendar.holidayList(start_date, end_date) for calendar in base_calendars]
        base_holidays = set(itertools.chain.from_iterable(base_holidays))
        for holiday in base_holidays:
            self.assertIn(holiday, joint_holidays)


if __name__ == "__main__":
    print("testing QuantLib", ql.__version__)
    unittest.main(verbosity=2)
