SPLIT_PROMPT = f"""
Segment the following sentence into individual facts:

Sentence: A most relaxing holiday spot..the staff worked so hard to repair the hurricane damage,but in such a unobtrusive way that one's stay there was not impaired in the least.The beach area was always tidy,ample beds and shade,good service at all times and the room was directly facing the sand...blissful.I confess I didn't want to do anything but take to my sunbed,read,swim,bask and take refreshment..I felt secure there at all times;there was always an attendant around even at night but again without having a strong-arm presence.The hotel guests were obviously having a good time;the friendly approach of the dining attendants ensured this at meal-times.The towels in the rooms were "sculptured" into an array of animals,with a new piece of artwork appearing at every room service.
Facts:
- The hotel was a most relaxing holiday spot.
- The staff worked very hard to repair the hurricane damage.
- The staff repaired the hurricane damage in an unobtrusive way.
- The staff repaired the hurricane damage without impairing anyone's stay
- The beach area was always tidy.
- The beach are had ample beds.
- The beach area was always tidy.
- There was good service at all times.
- The room was directly facing the sand.
- The room directly faced the sand so it was blissful.
- I didn't want to do anything but take to their sunbed, read, swim, bask, and take refreshment.
- I felt secure at all times.
- There was always an attendant around.
- There was an attendant around at night.
- The attendants were around without having a strong-arm presence.
- The friendly approach of the dining attendants at meal-times ensured that the hotel guests were having a good time.
- The towels in the room were sculptured into an array of animals.
- There was a new piece of artwork appearing at every room service.

Sentence: When we arrived early in the morning we had to wait for a few hours until checkin, but were still allowed to checkin around 9am which was quite a few hours before their official time.
Facts:
- We arrived early in the morning.
- We had to wait for a few hours until checkin.
- We were still allowed to checkin around 9am.
- We were allowed to checkin quite a few hours before their official time.


Sentence: Maybe this is the service you receive when you go to Flashy hotels and if this is the case then I am happily never going to a fancy hotel again - midrange hotels for me (better value for money)
Facts:
- Maybe this is the service you receive when you go to Flashy hotels.
- If this is the case, then I am happily never going to a fancy hotel again.
- Midrange hotels for me are better value for money.

Sentence: It sits on a hillside with 60 acres of garden to enjoy and offers fantasic views of the whole Bosphorus as well as being close to several restaurants and shopping.
Facts:
- It sits on a hillside.
- There are 60 acres of garden to enjoy.
- It offers fantastic views of the whole Bosphorus
- It is close to several restaurants.
- It is close to shopping.

Sentence: There are some really cool and handy features and amenities included in the rooms such as iPod docking station, a pressure shower, DVD player and television and Jacuzzi tub in the bathroom was great.
Facts:
- There are some really cool features.
- There are some really cool amenities.
- There are some really handy features.
- There are some really handy amenities.
- There is an iPod docking station.
- There is a pressure shower.
- There is a DVD player.
- The television in the bathroom was great.
- The Jacuzzi tub in the bathroom was great.

Sentence: This location is great if you're looking to access the great fun and action close-by, but with that comes hustle-and-bustle noise from the street the hotel is located on.
Facts:
- This location is great for the great fun and action close-by.
- This location is not great for the hustle-and-bustle noise from the street the hotel is located on.

Sentence: {{}}
Facts:\n"""
